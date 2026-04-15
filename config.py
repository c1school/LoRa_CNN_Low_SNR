"""프로젝트 전체 설정을 한 곳에 모아 둔 파일이다.

이 파일만 읽어도 아래 내용을 한눈에 파악할 수 있도록 구성한다.

1. 어떤 LoRa 프로파일을 실험할지
2. CNN 입력 feature bank를 얼마나 크게 만들지
3. 학습을 몇 epoch, 어떤 batch size로 돌릴지
4. calibration / test를 어떤 SNR 구간에서 할지
5. 하이브리드 정책을 어떤 기준으로 보정할지
6. 채널에 어떤 impairment를 넣을지

코드 다른 부분에서는 이 설정을 그대로 읽거나,
receiver_profile별 override를 기본값 위에 덮어써 사용한다.
"""


CFG = {
    # receiver_profiles:
    # 실제로 순회할 실험 프로파일 목록이다.
    # main.py는 이 리스트의 각 원소를 하나의 독립 실행 단위로 본다.
    "receiver_profiles": [
        {
            # name:
            # 그래프 파일명, CSV profile 컬럼, 체크포인트 이름에 들어가는 식별자다.
            "name": "sf7_bw125",
            # sf:
            # LoRa spreading factor다.
            # 심볼 후보 수 M = 2^SF 를 결정하므로 분류 문제 난이도와 심볼 길이가 바뀐다.
            "sf": 7,
            # bw:
            # LoRa 대역폭 [Hz]이다. 125e3은 125 kHz를 뜻한다.
            "bw": 125e3,
            # fs:
            # 시뮬레이터 내부 waveform 생성에 사용할 샘플링 주파수 [Hz]다.
            "fs": 1e6,
        },
        {
            "name": "sf8_bw125",
            "sf": 8,
            "bw": 125e3,
            "fs": 1e6,
            # training_overrides:
            # 이 프로파일에 한해서 기본 training 설정을 덮어쓴다.
            "training_overrides": {
                # train_batch_size:
                # 한 번의 optimizer step에서 동시에 학습할 샘플 수다.
                "train_batch_size": 32,
                # eval_batch_size:
                # validation / calibration / test 시 한 번에 처리할 샘플 수다.
                "eval_batch_size": 64,
                # num_epochs:
                # 전체 학습셋을 몇 바퀴 반복해서 학습할지 정한다.
                "num_epochs": 30,
            },
            # experiment_overrides:
            # 데이터 개수나 평가 범위를 이 프로파일에 맞게 별도로 조정할 때 쓴다.
            "experiment_overrides": {
                # train_samples:
                # 한 epoch 동안 온라인으로 생성할 학습 샘플 수다.
                "train_samples": 96000,
            },
            # model_overrides:
            # 이 프로파일에서만 CNN 크기를 키우거나 줄이고 싶을 때 쓴다.
            "model_overrides": {
                # width_scale:
                # 합성곱 채널 수 전체에 곱하는 배율이다.
                "width_scale": 1.25,
                # classifier_hidden:
                # classifier fully connected hidden 차원이다.
                "classifier_hidden": 384,
            },
        },
        {
            "name": "sf9_bw250",
            "sf": 9,
            "bw": 250e3,
            "fs": 2e6,
            "training_overrides": {
                # SF9는 입력이 커서 메모리 부담이 크므로 batch를 더 작게 잡는다.
                "train_batch_size": 8,
                "eval_batch_size": 16,
                "num_epochs": 40,
            },
            "experiment_overrides": {
                # 클래스 수가 크므로 학습 샘플 수도 더 늘린다.
                "train_samples": 192000,
            },
            "model_overrides": {
                # 큰 SF에서는 표현력을 더 주기 위해 모델 폭도 키운다.
                "width_scale": 1.5,
                "classifier_hidden": 512,
            },
        },
    ],

    # model:
    # CNN 구조의 기본값이다.
    # profile별 model_overrides가 있으면 그 값이 우선 적용된다.
    "model": {
        # width_scale:
        # stage_channels 전체에 곱해 실제 채널 수를 조절하는 전역 배율이다.
        "width_scale": 1.0,
        # classifier_hidden:
        # 마지막 분류기 hidden 차원 기본값이다.
        "classifier_hidden": 256,
        # dropout:
        # classifier에서 과적합을 줄이기 위한 dropout 비율이다.
        "dropout": 0.3,
        # stage_channels:
        # CNN feature extractor 4개 stage의 기본 출력 채널 수다.
        "stage_channels": [32, 64, 96, 128],
    },

    # feature_bank:
    # Default LoRa 결과를 CNN 입력으로 바꾸는 hypothesis bank 크기 설정이다.
    "feature_bank": {
        # patch_size:
        # 각 FFT bin 중심 주변으로 몇 개 bin을 함께 잘라 볼지 정한다.
        # patch_size가 5면 중심 bin 포함 5개 bin 에너지를 본다.
        "patch_size": 5,
        # cfo_steps:
        # CFO 가설 축을 몇 단계로 나눌지 정한다.
        "cfo_steps": 17,
        # to_steps:
        # timing offset 가설 축을 몇 단계로 나눌지 정한다.
        "to_steps": 9,
        # baseline_window:
        # grouped-bin classical score 계산 시 중심 bin 주변 몇 개를 합칠지 정한다.
        "baseline_window": 2,
    },

    # training:
    # 학습 루프 자체의 기본 하이퍼파라미터다.
    "training": {
        # train_batch_size:
        # 학습 미니배치 크기다.
        "train_batch_size": 64,
        # eval_batch_size:
        # validation / calibration / test 시 사용할 배치 크기다.
        "eval_batch_size": 128,
        # num_epochs:
        # 학습 데이터를 몇 바퀴 반복할지 정한다.
        "num_epochs": 20,
        # learning_rate:
        # Adam optimizer의 학습률이다.
        "learning_rate": 5e-4,
        # weight_decay:
        # L2 regularization 강도다.
        "weight_decay": 1e-4,
    },

    # experiment:
    # 데이터 개수와 평가 범위를 정의한다.
    "experiment": {
        # payload_symbols:
        # packet error rate 계산 시 한 packet을 몇 개 payload symbol로 볼지 정한다.
        "payload_symbols": 16,
        # train_samples:
        # 한 epoch에 온라인으로 생성해 학습할 샘플 수다.
        "train_samples": 48000,
        # val_packets:
        # validation용으로 미리 고정 생성할 packet 수다.
        "val_packets": 256,
        # calib_packets:
        # 하이브리드 threshold/bin policy를 맞추기 위해 SNR별로 생성할 packet 수다.
        "calib_packets": 300,
        # test_packets:
        # seen / unseen 평가에서 SNR별로 생성할 packet 수다.
        "test_packets": 500,
        # seeds:
        # 반복 실행에 사용할 난수 시드 목록이다.
        "seeds": [2024, 2025, 2026],
        # train_snr_range:
        # 온라인 학습 중 SNR을 어느 범위에서 샘플링할지 정한다.
        "train_snr_range": (-23.0, -5.0),
        # test_snrs:
        # calibration / seen / unseen 평가에 사용할 SNR 지점 목록이다.
        "test_snrs": list(range(-21, 1)),
    },

    # hybrid:
    # Default LoRa와 CNN을 어떻게 섞을지 정하는 정책 설정이다.
    "hybrid": {
        # confidence_type:
        # Default LoRa 결과의 신뢰도를 어떤 방식으로 계산할지 정한다.
        "confidence_type": "ratio",
        # global_threshold_grid:
        # global threshold 후보를 몇 단계로 스윕할지 정한다.
        "global_threshold_grid": 81,
        # confidence_bins:
        # confidence-bin policy에서 confidence 축을 몇 구간으로 나눌지 정한다.
        "confidence_bins": 10,
        # ser_tolerance:
        # Full CNN 대비 허용할 SER 손실 한계다.
        "ser_tolerance": 0.005,
        # per_tolerance:
        # Full CNN 대비 허용할 PER 손실 한계다.
        "per_tolerance": 0.01,
    },

    # benchmark:
    # 상대 추론 시간 측정을 위한 설정이다.
    "benchmark": {
        # warmup:
        # 실제 측정 전에 cache와 커널을 안정화하기 위한 예열 횟수다.
        "warmup": 2,
        # repeats:
        # 실제 측정을 몇 번 반복할지 정한다.
        "repeats": 6,
        # batch_size:
        # benchmark용 waveform 수다.
        "batch_size": 64,
    },

    # artifacts:
    # best 모델 저장 방식과 경로를 정의한다.
    "artifacts": {
        # save_best_weights:
        # validation loss 기준 best 가중치를 저장할지 여부다.
        "save_best_weights": True,
        # save_best_checkpoint:
        # optimizer 상태까지 포함한 checkpoint를 저장할지 여부다.
        "save_best_checkpoint": True,
        # weights_dir:
        # best weights 저장 폴더다.
        "weights_dir": "artifacts/weights",
        # checkpoints_dir:
        # best checkpoint 저장 폴더다.
        "checkpoints_dir": "artifacts/checkpoints",
    },

    # recorded_eval_npz:
    # 외부에서 수집한 npz 파일 평가에 사용할 경로 목록이다.
    "recorded_eval_npz": [],

    # channel_profiles:
    # 학습과 평가에 사용할 impairment 분포다.
    "channel_profiles": {
        "train": {
            # max_cfo_bins:
            # CFO를 LoRa bin 간격 기준으로 얼마나 크게 허용할지 정한다.
            "max_cfo_bins": 0.45,
            # max_to_samples:
            # 정수 샘플 단위 timing offset 최대값이다.
            "max_to_samples": 4,
            # max_to_symbol_fraction:
            # 심볼 길이 대비 timing offset 비율 표현이다.
            "max_to_symbol_fraction": 4 / 1024,
            # max_fractional_to_samples:
            # 정수 샘플보다 작은 fractional timing offset 최대값이다.
            "max_fractional_to_samples": 0.35,
            # max_paths:
            # multipath 경로 수 최대값이다.
            "max_paths": 4,
            # max_delay_samples:
            # 추가 경로 delay 최대 샘플 수다.
            "max_delay_samples": 18,
            # max_delay_symbol_fraction:
            # 심볼 길이 대비 delay spread 비율 표현이다.
            "max_delay_symbol_fraction": 18 / 1024,
            # delay_decay:
            # 지연된 경로로 갈수록 gain을 얼마나 빠르게 줄일지 정한다.
            "delay_decay": 5.0,
            # extra_path_prob:
            # direct path 외에 추가 경로가 실제로 생길 확률이다.
            "extra_path_prob": 0.75,
            # phase_noise_std_range:
            # phase noise random walk 표준편차 범위다.
            "phase_noise_std_range": (0.0004, 0.0018),
            # tone_interference_prob:
            # narrowband tone interference 삽입 확률이다.
            "tone_interference_prob": 0.20,
            # tone_inr_db_range:
            # interference 전력 범위를 INR[dB]로 정한다.
            "tone_inr_db_range": (-12.0, 4.0),
        },
        "seen_eval": {
            # seen_eval:
            # train과 같은 분포의 채널에서 일반화 성능을 보는 설정이다.
            "max_cfo_bins": 0.45,
            "max_to_samples": 4,
            "max_to_symbol_fraction": 4 / 1024,
            "max_fractional_to_samples": 0.35,
            "max_paths": 4,
            "max_delay_samples": 18,
            "max_delay_symbol_fraction": 18 / 1024,
            "delay_decay": 5.0,
            "extra_path_prob": 0.75,
            "phase_noise_std_range": (0.0004, 0.0018),
            "tone_interference_prob": 0.20,
            "tone_inr_db_range": (-12.0, 4.0),
        },
        "unseen_eval": {
            # unseen_eval:
            # train보다 더 강한 impairment를 넣어 stress test처럼 평가하는 설정이다.
            "max_cfo_bins": 0.75,
            "max_to_samples": 8,
            "max_to_symbol_fraction": 8 / 1024,
            "max_fractional_to_samples": 0.70,
            "max_paths": 5,
            "max_delay_samples": 28,
            "max_delay_symbol_fraction": 28 / 1024,
            "delay_decay": 9.0,
            "extra_path_prob": 0.90,
            "phase_noise_std_range": (0.0015, 0.0040),
            "tone_interference_prob": 0.45,
            "tone_inr_db_range": (-6.0, 9.0),
        },
    },
}
