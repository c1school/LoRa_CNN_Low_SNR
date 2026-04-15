"""프로파일/시드 단위 실행 로직을 분리한 모듈이다.

main.py는 이 모듈을 호출해 실제 학습/평가를 수행한다.
즉 이 파일은 "한 profile, 한 seed를 끝까지 돌리는 절차"를 담고 있다.
"""

import os

import torch
from torch.utils.data import DataLoader

from dataset import (
    OnlineParametersDataset,
    create_fixed_waveform_dataset,
    create_fixed_waveform_range_dataset,
)
from evaluation import (
    benchmark_receivers,
    calibrate_confidence_bin_policy_from_outputs,
    calibrate_global_threshold_from_outputs,
    collect_receiver_outputs,
    summarize_outputs,
)
from models import Hypothesis2DCNN
from simulator import GPUOnlineSimulator
from training import train_online_model
from utils import count_trainable_parameters, merge_config, set_seed


def build_profile_runtime_configs(
    receiver_profile,
    base_experiment_cfg,
    base_train_cfg,
    base_feature_cfg,
    base_model_cfg,
    base_benchmark_cfg,
):
    """프로파일별 override를 기본 설정과 병합해 실제 실행 설정을 만든다."""

    # merge_config는 base_config 위에 overrides를 덮어쓴 새 dict를 반환한다.
    experiment_cfg = merge_config(base_experiment_cfg, receiver_profile.get("experiment_overrides"))
    train_cfg = merge_config(base_train_cfg, receiver_profile.get("training_overrides"))
    feature_cfg = merge_config(base_feature_cfg, receiver_profile.get("feature_bank_overrides"))
    model_cfg = merge_config(base_model_cfg, receiver_profile.get("model_overrides"))
    benchmark_cfg = merge_config(base_benchmark_cfg, receiver_profile.get("benchmark_overrides"))
    return experiment_cfg, train_cfg, feature_cfg, model_cfg, benchmark_cfg


def _build_model(simulator, feature_cfg, model_cfg):
    """현재 프로파일 설정에 맞는 CNN 모델을 만든다."""

    return Hypothesis2DCNN(
        # num_classes:
        # 최종 분류해야 하는 심볼 후보 수다.
        num_classes=simulator.M,
        # num_hypotheses:
        # CFO 가설 수 * timing offset 가설 수다.
        num_hypotheses=feature_cfg["cfo_steps"] * feature_cfg["to_steps"],
        # num_bins:
        # 가설 하나당 보는 FFT patch 길이다.
        num_bins=simulator.M * feature_cfg["patch_size"],
        # in_channels:
        # 현재 feature bank는 실수부/허수부 또는 유사 2채널 표현을 사용하므로 2다.
        in_channels=2,
        stage_channels=model_cfg["stage_channels"],
        classifier_hidden=model_cfg["classifier_hidden"],
        dropout=model_cfg["dropout"],
        width_scale=model_cfg["width_scale"],
    ).to(simulator.device)


def _load_model_from_checkpoint(model, checkpoint_path, device):
    """저장된 체크포인트나 가중치 파일에서 모델 파라미터만 읽어 온다.

    현재 저장 포맷은 두 가지다.

    - `*_best_checkpoint.pt`
      optimizer 상태, epoch, best loss, metadata까지 함께 들어 있는 파일
    - `*_best_weights.pth`
      모델 가중치와 일부 메타데이터만 들어 있는 파일

    두 파일 모두 `model_state_dict` 키를 포함하므로,
    우선 그 키를 찾고 없으면 파일 자체를 state_dict로 간주한다.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return payload


def _build_datasets(simulator, experiment_cfg, channel_profiles, seed):
    """validation, calibration, seen test, unseen test 데이터셋을 만든다."""

    # ds_val:
    # train SNR 범위를 대표하는 validation waveform 데이터셋이다.
    ds_val = create_fixed_waveform_range_dataset(
        simulator,
        num_packets=experiment_cfg["val_packets"],
        snr_range=experiment_cfg["train_snr_range"],
        channel_profile=channel_profiles["train"],
        seed=seed,
        experiment_cfg=experiment_cfg,
    )

    # ds_calib:
    # 하이브리드 threshold/bin policy를 맞추기 위한 calibration 데이터셋이다.
    ds_calib = create_fixed_waveform_dataset(
        simulator,
        num_packets_per_snr=experiment_cfg["calib_packets"],
        snr_list=experiment_cfg["test_snrs"],
        channel_profile=channel_profiles["seen_eval"],
        seed=seed + 1,
        experiment_cfg=experiment_cfg,
    )

    # ds_test_seen:
    # train과 같은 분포의 seen 채널 평가 데이터셋이다.
    ds_test_seen = create_fixed_waveform_dataset(
        simulator,
        num_packets_per_snr=experiment_cfg["test_packets"],
        snr_list=experiment_cfg["test_snrs"],
        channel_profile=channel_profiles["seen_eval"],
        seed=seed + 2,
        experiment_cfg=experiment_cfg,
    )

    # ds_test_unseen:
    # train보다 더 harsh한 분포의 unseen 채널 평가 데이터셋이다.
    ds_test_unseen = create_fixed_waveform_dataset(
        simulator,
        num_packets_per_snr=experiment_cfg["test_packets"],
        snr_list=experiment_cfg["test_snrs"],
        channel_profile=channel_profiles["unseen_eval"],
        seed=seed + 3,
        experiment_cfg=experiment_cfg,
    )
    return ds_val, ds_calib, ds_test_seen, ds_test_unseen


def _build_train_loader(simulator, experiment_cfg, train_cfg, channel_profiles, pin_memory):
    """온라인 학습용 파라미터 데이터셋과 DataLoader를 만든다."""

    return DataLoader(
        OnlineParametersDataset(
            # simulator.M:
            # label 범위를 0 ~ M-1로 정하기 위해 심볼 후보 수를 전달한다.
            simulator.M,
            # train_samples:
            # 한 epoch에 몇 개의 온라인 샘플을 생성할지 정한다.
            experiment_cfg["train_samples"],
            # train_snr_range:
            # 온라인 학습 시 SNR 샘플링 범위다.
            experiment_cfg["train_snr_range"],
            # max_cfo_bins:
            # 학습 시 CFO 샘플링 범위를 정한다.
            channel_profiles["train"]["max_cfo_bins"],
            # simulator.bw:
            # bin 단위 CFO 범위를 실제 Hz로 환산할 때 사용한다.
            simulator.bw,
        ),
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        pin_memory=pin_memory,
    )


def _build_val_loader(ds_val, train_cfg, pin_memory):
    """고정 validation waveform 데이터셋용 DataLoader를 만든다."""

    return DataLoader(
        ds_val,
        batch_size=train_cfg["eval_batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
    )


def run_profile_seed(
    receiver_profile,
    seed,
    *,
    artifact_cfg,
    hybrid_cfg,
    channel_profiles,
    experiment_cfg,
    train_cfg,
    feature_cfg,
    model_cfg,
    benchmark_cfg,
    pin_memory,
):
    """프로파일/시드 조합 1개를 끝까지 실행하고 raw 결과를 반환한다."""

    # set_seed:
    # 현재 profile/seed 실행의 난수 상태를 고정한다.
    set_seed(seed)

    # simulator:
    # waveform 생성, 채널 적용, feature bank 추출을 담당한다.
    simulator = GPUOnlineSimulator(
        sf=receiver_profile["sf"],
        bw=receiver_profile["bw"],
        fs=receiver_profile["fs"],
        device="cuda" if pin_memory else "cpu",
    )

    # profile 설정에 맞는 CNN 모델을 만든다.
    model = _build_model(simulator, feature_cfg, model_cfg)

    # validation / calibration / test용 고정 데이터셋을 만든다.
    ds_val, ds_calib, ds_test_seen, ds_test_unseen = _build_datasets(
        simulator,
        experiment_cfg,
        channel_profiles,
        seed,
    )

    # train은 online parameter dataset, val은 fixed waveform dataset을 사용한다.
    dl_train = _build_train_loader(simulator, experiment_cfg, train_cfg, channel_profiles, pin_memory)
    dl_val = _build_val_loader(ds_val, train_cfg, pin_memory)

    checkpoint_path = receiver_profile.get("checkpoint_path")

    if checkpoint_path:
        # checkpoint_path가 지정되면 이미 저장된 모델을 불러와
        # 재학습 없이 calibration / evaluation / plotting 단계만 수행한다.
        _load_model_from_checkpoint(model, checkpoint_path, simulator.device)
        print(f"-> Loaded checkpoint for evaluation only: {checkpoint_path}")
    else:
        # train_online_model:
        # 학습을 수행하고 best validation checkpoint를 복원한 모델을 반환한다.
        model = train_online_model(
            model,
            simulator,
            dl_train,
            dl_val,
            channel_profile=channel_profiles["train"],
            train_cfg=train_cfg,
            feature_cfg=feature_cfg,
            artifact_cfg=artifact_cfg,
            run_name=f"{receiver_profile['name']}_seed{seed}",
            metadata={
                "profile": receiver_profile["name"],
                "seed": seed,
                "sf": receiver_profile["sf"],
                "bw": receiver_profile["bw"],
                "fs": receiver_profile["fs"],
                "train_cfg": train_cfg,
                "experiment_cfg": experiment_cfg,
                "feature_cfg": feature_cfg,
                "model_cfg": model_cfg,
                "hybrid_cfg": hybrid_cfg,
                "train_channel_profile": simulator.resolve_channel_profile(channel_profiles["train"]),
            },
        )

    # collect_receiver_outputs:
    # calibration 데이터에 대해 Default LoRa / Full CNN / confidence 등을 한 번 모아 둔다.
    calib_outputs = collect_receiver_outputs(
        model,
        simulator,
        ds_calib,
        channel_profiles["seen_eval"],
        feature_cfg=feature_cfg,
        eval_batch_size=train_cfg["eval_batch_size"],
        hybrid_cfg=hybrid_cfg,
    )

    # global_policy:
    # 모든 SNR에서 공통으로 쓰는 단일 threshold 정책이다.
    global_policy = calibrate_global_threshold_from_outputs(
        calib_outputs,
        hybrid_cfg=hybrid_cfg,
        payload_symbols=experiment_cfg["payload_symbols"],
    )

    # bin_policy:
    # confidence 구간별로 threshold를 다르게 두는 정책이다.
    bin_policy = calibrate_confidence_bin_policy_from_outputs(
        calib_outputs,
        hybrid_cfg=hybrid_cfg,
        payload_symbols=experiment_cfg["payload_symbols"],
    )

    # seen / unseen 평가 데이터에서도 같은 형태의 출력을 모은다.
    test_seen_outputs = collect_receiver_outputs(
        model,
        simulator,
        ds_test_seen,
        channel_profiles["seen_eval"],
        feature_cfg=feature_cfg,
        eval_batch_size=train_cfg["eval_batch_size"],
        hybrid_cfg=hybrid_cfg,
    )
    test_unseen_outputs = collect_receiver_outputs(
        model,
        simulator,
        ds_test_unseen,
        channel_profiles["unseen_eval"],
        feature_cfg=feature_cfg,
        eval_batch_size=train_cfg["eval_batch_size"],
        hybrid_cfg=hybrid_cfg,
    )

    # summarize_outputs:
    # 각 policy와 각 채널 분포에 대해 SNR별 SER/PER/util 통계를 만든다.
    results = {
        "global_seen": summarize_outputs(
            test_seen_outputs,
            global_policy,
            hybrid_cfg=hybrid_cfg,
            payload_symbols=experiment_cfg["payload_symbols"],
        ),
        "bin_seen": summarize_outputs(
            test_seen_outputs,
            bin_policy,
            hybrid_cfg=hybrid_cfg,
            payload_symbols=experiment_cfg["payload_symbols"],
        ),
        "global_unseen": summarize_outputs(
            test_unseen_outputs,
            global_policy,
            hybrid_cfg=hybrid_cfg,
            payload_symbols=experiment_cfg["payload_symbols"],
        ),
        "bin_unseen": summarize_outputs(
            test_unseen_outputs,
            bin_policy,
            hybrid_cfg=hybrid_cfg,
            payload_symbols=experiment_cfg["payload_symbols"],
        ),
    }

    # benchmark는 평가 SNR 중앙값 지점 하나를 골라 상대 추론 시간을 측정한다.
    benchmark_snr = experiment_cfg["test_snrs"][len(experiment_cfg["test_snrs"]) // 2]
    latency_row = {
        "profile": receiver_profile["name"],
        "seed": seed,
        # params:
        # 현재 모델의 학습 가능한 파라미터 수다.
        "params": count_trainable_parameters(model),
        # num_hypotheses:
        # CFO 가설 수와 timing 가설 수를 곱한 총 hypothesis 수다.
        "num_hypotheses": feature_cfg["cfo_steps"] * feature_cfg["to_steps"],
        # feature_elements:
        # CNN 입력 feature bank 전체 원소 수다.
        "feature_elements": feature_cfg["cfo_steps"] * feature_cfg["to_steps"] * simulator.M * feature_cfg["patch_size"],
        **benchmark_receivers(
            model,
            simulator,
            ds_test_seen[benchmark_snr],
            channel_profiles["seen_eval"],
            bin_policy,
            benchmark_cfg=benchmark_cfg,
            feature_cfg=feature_cfg,
            hybrid_cfg=hybrid_cfg,
        ),
    }

    # run_rows:
    # profile/seed 실행 결과를 summary 생성용 row 형태로 펼쳐 담는다.
    run_rows = []
    for eval_type, eval_stats in results.items():
        for snr in experiment_cfg["test_snrs"]:
            run_rows.append(
                {
                    "profile": receiver_profile["name"],
                    "seed": seed,
                    "snr": snr,
                    "type": eval_type,
                    **eval_stats[snr],
                }
            )

    return run_rows, latency_row
