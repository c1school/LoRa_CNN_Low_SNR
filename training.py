"""온라인 합성 채널을 사용해 CNN을 학습시키는 학습 루프 파일이다."""

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
    """온라인 합성 데이터로 모델을 학습하고, 최적 체크포인트를 복원해 반환한다."""

    train_cfg = CFG["training"] if train_cfg is None else train_cfg
    feature_cfg = CFG["feature_bank"] if feature_cfg is None else feature_cfg
    artifact_cfg = CFG["artifacts"] if artifact_cfg is None else artifact_cfg

    num_epochs = train_cfg["num_epochs"] if num_epochs is None else num_epochs
    lr = train_cfg["learning_rate"] if lr is None else lr
    weight_decay = train_cfg["weight_decay"] if weight_decay is None else weight_decay

    device = simulator.device
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    metadata = {} if metadata is None else metadata
    weights_path = None
    checkpoint_path = None

    if run_name is not None:
        if artifact_cfg.get("save_best_weights", False):
            os.makedirs(artifact_cfg["weights_dir"], exist_ok=True)
            weights_path = os.path.join(artifact_cfg["weights_dir"], f"{run_name}_best_weights.pth")
        if artifact_cfg.get("save_best_checkpoint", False):
            os.makedirs(artifact_cfg["checkpoints_dir"], exist_ok=True)
            checkpoint_path = os.path.join(artifact_cfg["checkpoints_dir"], f"{run_name}_best_checkpoint.pt")

    print(f"\nTraining 2D CNN on {device} for {num_epochs} epochs")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for labels, snrs, cfos in train_loader:
            labels = labels.to(device, non_blocking=True)
            snrs = snrs.to(device, non_blocking=True)
            cfos = cfos.to(device, non_blocking=True)

            # 매 배치마다 새로운 채널 상태를 샘플링해 데이터 다양성을 높인다.
            channel_state = simulator.sample_channel_state(labels.size(0), channel_profile)
            rx_signals = simulator.generate_batch(
                labels,
                snrs,
                cfos,
                channel_state=channel_state,
            )
            features = simulator.extract_multi_hypothesis_bank(
                rx_signals,
                helper=helper,
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for labels, rx_signals in val_loader:
                labels = labels.to(device, non_blocking=True)
                rx_signals = rx_signals.to(device, non_blocking=True)
                features = simulator.extract_multi_hypothesis_bank(
                    rx_signals,
                    helper=helper,
                )

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = (correct / max(total, 1)) * 100.0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = {name: tensor.cpu().clone() for name, tensor in model.state_dict().items()}

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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

    if best_model_state is None:
        raise RuntimeError("Model training did not produce a checkpoint.")

    print(f"-> Restoring best checkpoint (val loss {best_val_loss:.4f})")
    model.load_state_dict(best_model_state)
    if weights_path is not None:
        print(f"-> Saved best weights: {weights_path}")
    if checkpoint_path is not None:
        print(f"-> Saved best checkpoint: {checkpoint_path}")
    return model
