"""온라인으로 합성한 LoRa 채널 샘플을 사용해 CNN을 학습시키는 파일이다.

핵심 아이디어는 다음과 같다.

1. train_loader는 waveform 자체를 담고 있지 않는다.
2. 대신 `(label, SNR, CFO)` 같은 파라미터만 넘긴다.
3. training loop 안에서 simulator가 실제 waveform을 실시간으로 합성한다.
4. 그 waveform을 hypothesis feature bank로 바꿔 CNN에 넣는다.
5. validation은 반대로 고정 waveform 데이터셋을 사용해 epoch 간 비교를 안정적으로 한다.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from utils import get_max_cfo_hz, move_to_cpu


def train_online_model(
    model,
    simulator,
    train_loader,
    val_loader,
    channel_profile,
    num_epochs=None,
    lr=None,
    weight_decay=None,
    train_cfg=None,
    feature_cfg=None,
    artifact_cfg=None,
    run_name: str = None,
    metadata: dict = None,
):
    """모델을 학습시키고 best validation checkpoint를 복원한 상태로 반환한다.

    입력 인자의 역할은 아래와 같다.

    - model:
      실제로 학습할 CNN 모델이다.
    - simulator:
      waveform 생성, feature bank 추출, hypothesis helper 준비를 담당하는 객체다.
    - train_loader:
      `(label, SNR, CFO)`를 공급하는 온라인 학습용 DataLoader다.
    - val_loader:
      `(label, rx_waveform)`를 공급하는 고정 validation DataLoader다.
    - channel_profile:
      train용 impairment 분포를 뜻한다.
    - train_cfg:
      batch size, epoch 수, learning rate 같은 학습 하이퍼파라미터 묶음이다.
    - feature_cfg:
      feature bank 크기와 hypothesis 개수를 정하는 설정이다.
    - artifact_cfg:
      best weights / checkpoint 저장 여부와 저장 경로를 정하는 설정이다.
    - run_name:
      저장 파일 이름 접두어다.
    - metadata:
      체크포인트 안에 같이 저장할 실행 메타데이터다.
    """

    # train_cfg, feature_cfg, artifact_cfg를 따로 넘기지 않으면
    # config.py에 있는 기본값을 그대로 사용한다.
    train_cfg = CFG["training"] if train_cfg is None else train_cfg
    feature_cfg = CFG["feature_bank"] if feature_cfg is None else feature_cfg
    artifact_cfg = CFG["artifacts"] if artifact_cfg is None else artifact_cfg

    # 함수 인자로 num_epochs / lr / weight_decay를 직접 넘기면 그 값을 우선 사용한다.
    # 그렇지 않으면 train_cfg 안의 기본값을 사용한다.
    num_epochs = train_cfg["num_epochs"] if num_epochs is None else num_epochs
    lr = train_cfg["learning_rate"] if lr is None else lr
    weight_decay = train_cfg["weight_decay"] if weight_decay is None else weight_decay

    # simulator가 이미 CPU 또는 GPU device를 결정했으므로
    # 모델도 동일한 장치로 올린다.
    device = simulator.device
    model.to(device)

    # 현재 문제는 "심볼 클래스 분류"이므로 CrossEntropyLoss를 사용한다.
    criterion = nn.CrossEntropyLoss()

    # optimizer는 Adam을 사용한다.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train과 validation에서 매 배치마다 공통으로 사용할 hypothesis helper를 미리 만든다.
    # helper에는 CFO/TO 가설 보정 항, patch index 같은 반복 계산용 캐시가 들어간다.
    resolved_profile = simulator.resolve_channel_profile(channel_profile)
    max_cfo_hz = get_max_cfo_hz(simulator, channel_profile)
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(
        max_cfo_hz,
        resolved_profile["max_to_samples"],
        feature_cfg["cfo_steps"],
        feature_cfg["to_steps"],
    )
    helper = simulator.prepare_hypothesis_helper(
        cfo_grid,
        to_grid,
        feature_cfg["patch_size"],
    )

    # best_* 변수들은 validation loss 기준으로 가장 좋은 모델 상태를 추적한다.
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    metadata = {} if metadata is None else metadata
    weights_path = None
    checkpoint_path = None

    # run_name이 있으면 현재 실행의 best 모델을 파일로 저장할 경로를 만든다.
    if run_name is not None:
        if artifact_cfg.get("save_best_weights", False):
            os.makedirs(artifact_cfg["weights_dir"], exist_ok=True)
            weights_path = os.path.join(artifact_cfg["weights_dir"], f"{run_name}_best_weights.pth")
        if artifact_cfg.get("save_best_checkpoint", False):
            os.makedirs(artifact_cfg["checkpoints_dir"], exist_ok=True)
            checkpoint_path = os.path.join(artifact_cfg["checkpoints_dir"], f"{run_name}_best_checkpoint.pt")

    print(f"\nTraining 2D CNN on {device} for {num_epochs} epochs")

    # epoch 루프:
    # 전체 train_loader를 num_epochs번 반복하면서 학습한다.
    for epoch in range(num_epochs):
        # -------------------------
        # 1. 학습 단계
        # -------------------------
        model.train()
        train_loss = 0.0

        for labels, snrs, cfos in train_loader:
            # labels:
            # 각 샘플의 정답 심볼 인덱스다.
            labels = labels.to(device, non_blocking=True)
            # snrs:
            # 각 샘플에 적용할 SNR 값이다.
            snrs = snrs.to(device, non_blocking=True)
            # cfos:
            # 각 샘플에 적용할 CFO 값이다.
            cfos = cfos.to(device, non_blocking=True)

            # channel_state:
            # multipath, timing offset, phase noise, tone interference 등
            # 현재 배치에 적용할 impairment 상태를 샘플링한다.
            channel_state = simulator.sample_channel_state(labels.size(0), channel_profile)

            # generate_batch:
            # labels / SNR / CFO / channel_state를 받아 실제 IQ waveform을 만든다.
            rx_signals = simulator.generate_batch(
                labels,
                snrs,
                cfos,
                channel_state=channel_state,
            )

            # extract_multi_hypothesis_bank:
            # IQ waveform을 여러 CFO/TO 가설에 대해 dechirp/FFT한 결과로 바꾼다.
            # CNN은 이 feature bank를 입력으로 받는다.
            features = simulator.extract_multi_hypothesis_bank(
                rx_signals,
                helper=helper,
            )

            # 이전 배치에서 남아 있는 gradient를 먼저 비운다.
            optimizer.zero_grad(set_to_none=True)

            # 모델 출력 shape은 [batch, num_classes]이고,
            # 각 클래스에 대한 logits를 담는다.
            outputs = model(features)

            # labels와 logits를 비교해 분류 손실을 계산한다.
            loss = criterion(outputs, labels)

            # backward:
            # 현재 손실 기준으로 gradient를 계산한다.
            loss.backward()

            # step:
            # 계산된 gradient로 파라미터를 실제 갱신한다.
            optimizer.step()

            train_loss += loss.item()

        # -------------------------
        # 2. validation 단계
        # -------------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for labels, rx_signals in val_loader:
                # validation용 waveform은 미리 고정 생성돼 있으므로
                # epoch별 성능 비교가 더 안정적이다.
                labels = labels.to(device, non_blocking=True)
                rx_signals = rx_signals.to(device, non_blocking=True)

                features = simulator.extract_multi_hypothesis_bank(
                    rx_signals,
                    helper=helper,
                )

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # argmax로 최종 예측 심볼을 만든다.
                predictions = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = (correct / max(total, 1)) * 100.0

        # validation loss가 갱신되면 현재 상태를 best 모델로 기록한다.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            # state_dict를 CPU clone으로 저장해 두면
            # 학습 장치 상태와 무관하게 안전하게 best 모델을 복원할 수 있다.
            best_model_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}

            # weights 파일:
            # 모델 가중치만 저장한다.
            if weights_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_model_state,
                        "epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "metadata": metadata,
                    },
                    weights_path,
                )

            # checkpoint 파일:
            # 모델 가중치뿐 아니라 optimizer 상태와 메타데이터까지 함께 저장한다.
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_model_state,
                        "optimizer_state_dict": move_to_cpu(optimizer.state_dict()),
                        "epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "metadata": metadata,
                    },
                    checkpoint_path,
                )

        # 모든 epoch를 다 찍으면 로그가 너무 길어지므로
        # 첫 epoch와 5 epoch 간격에서만 출력한다.
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

    # best checkpoint가 한 번도 갱신되지 않았다면 학습이 비정상 종료된 것으로 본다.
    if best_model_state is None:
        raise RuntimeError("Model training did not produce a checkpoint.")

    # 마지막 epoch 상태가 아니라 best validation loss 상태를 복원해 반환한다.
    print(f"-> Restoring best checkpoint (val loss {best_val_loss:.4f})")
    model.load_state_dict(best_model_state)
    if weights_path is not None:
        print(f"-> Saved best weights: {weights_path}")
    if checkpoint_path is not None:
        print(f"-> Saved best checkpoint: {checkpoint_path}")
    return model
