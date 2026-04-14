"""실험 전체에서 사용하는 기본 설정을 한 곳에 모아 둔 파일이다.

이 파일은 다음 정보를 정의한다.

- 어떤 SF / BW 조합을 실험할 것인지
- feature bank를 어떤 해상도로 만들 것인지
- 학습 배치 크기와 epoch 수는 얼마인지
- 평가할 SNR 구간은 어디까지인지
- 하이브리드 정책 보정 시 허용 오차는 얼마인지
- 채널에 어떤 impairment를 어느 정도까지 넣을 것인지

실행 중에는 `main.py`와 `colab_run.py`가 이 설정을 그대로 사용하거나,
프로파일별 오버라이드를 덧씌워서 사용한다.
"""


CFG = {
    # 수신기 프로파일 목록이다.
    # 각 항목은 하나의 독립적인 실험 동작점으로 취급된다.
    "receiver_profiles": [
        {"name": "sf7_bw125", "sf": 7, "bw": 125e3, "fs": 1e6},
        {
            "name": "sf8_bw125",
            "sf": 8,
            "bw": 125e3,
            "fs": 1e6,
            # 큰 SF에서 학습 난이도가 더 높으므로 기본값보다 학습 예산을 늘린다.
            "training_overrides": {
                "train_batch_size": 32,
                "eval_batch_size": 64,
                "num_epochs": 30,
            },
            "experiment_overrides": {
                "train_samples": 96000,
            },
            # 분류 클래스 수가 늘어나므로 모델 폭도 약간 키운다.
            "model_overrides": {
                "width_scale": 1.25,
                "classifier_hidden": 384,
            },
        },
        {
            "name": "sf9_bw250",
            "sf": 9,
            "bw": 250e3,
            "fs": 2e6,
            # SF9는 feature 크기와 분류 난이도가 모두 커서 배치를 더 작게 잡는다.
            "training_overrides": {
                "train_batch_size": 8,
                "eval_batch_size": 16,
                "num_epochs": 40,
            },
            "experiment_overrides": {
                "train_samples": 192000,
            },
            "model_overrides": {
                "width_scale": 1.5,
                "classifier_hidden": 512,
            },
        },
    ],
    # CNN 구조의 기본 크기다.
    # 프로파일별 override가 있으면 그 값을 우선 사용한다.
    "model": {
        "width_scale": 1.0,
        "classifier_hidden": 256,
        "dropout": 0.3,
        "stage_channels": [32, 64, 96, 128],
    },
    # 다중 가설 feature bank를 만들 때 사용하는 설정이다.
    "feature_bank": {
        "patch_size": 5,
        "cfo_steps": 17,
        "to_steps": 9,
        "baseline_window": 2,
    },
    # 학습 루프 기본 설정이다.
    "training": {
        "train_batch_size": 64,
        "eval_batch_size": 128,
        "num_epochs": 20,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
    },
    # 데이터 개수와 평가 SNR 구간을 정의한다.
    "experiment": {
        "payload_symbols": 16,
        "train_samples": 48000,
        "val_packets": 256,
        "calib_packets": 300,
        "test_packets": 500,
        "seeds": [2024, 2025, 2026],
        "train_snr_range": (-23.0, -5.0),
        # 그래프를 더 촘촘하게 보기 위해 -21 dB부터 0 dB까지 1 dB 간격으로 평가한다.
        "test_snrs": list(range(-21, 1)),
    },
    # 하이브리드 정책 보정에 사용되는 설정이다.
    "hybrid": {
        "confidence_type": "ratio",
        "global_threshold_grid": 81,
        "confidence_bins": 10,
        "ser_tolerance": 0.005,
        "per_tolerance": 0.01,
    },
    # 상대적인 추론 지연시간을 재는 데 사용하는 설정이다.
    "benchmark": {
        "warmup": 2,
        "repeats": 6,
        "batch_size": 64,
    },
    # 학습 중 최적 모델을 저장하는 경로와 저장 여부를 정의한다.
    "artifacts": {
        "save_best_weights": True,
        "save_best_checkpoint": True,
        "weights_dir": "artifacts/weights",
        "checkpoints_dir": "artifacts/checkpoints",
    },
    # 외부에서 측정한 IQ 데이터를 평가에 쓰고 싶을 때 `.npz` 경로를 넣을 수 있다.
    "recorded_eval_npz": [],
    # 채널 프로파일 정의다.
    # train / seen_eval / unseen_eval 세 가지 분포를 나누어 사용한다.
    "channel_profiles": {
        "train": {
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
        "seen_eval": {
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
