import torch
import torch.nn as nn
import torch.optim as optim
from config import CFG



def train_online_model(model, simulator, train_loader, val_loader, max_cfo_hz, num_epochs=20, lr=0.0005):
    """
    온라인 방식으로 모델을 학습하는 함수이다.

    이 함수의 가장 큰 특징은 학습용 파형을 미리 저장하지 않고,
    매 배치마다 시뮬레이터가 새로운 파형을 즉석에서 생성한다는 점이다.
    즉, 학습 데이터는 "파일"이 아니라 "조건(label, SNR, CFO)"로 주어지고,
    실제 IQ 파형은 GPU에서 바로 만들어진다.

    처리 순서는 다음과 같다.
    1) train_loader에서 label, SNR, CFO를 가져온다.
    2) simulator로 수신 파형을 생성한다.
    3) 다중 가설 2차원 특징맵을 만든다.
    4) 모델이 심볼을 분류하도록 학습한다.
    5) 고정 validation dataset으로 성능을 점검한다.
    6) validation loss가 가장 좋았던 모델을 최종 모델로 복원한다.
    """

    device = simulator.device
    model.to(device)

    # 다중 클래스 심볼 분류이므로 CrossEntropyLoss를 사용한다.
    criterion = nn.CrossEntropyLoss()

    # Adam 옵티마이저를 사용한다.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n2차원 CNN 온라인 학습 시작. Device: {device}, Epochs: {num_epochs}")

    best_val_loss = float("inf")
    best_model_state = None

    # 학습 동안 계속 사용할 CFO / Timing 가설 격자를 미리 만든다.
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(
        max_cfo_hz,
        CFG["max_to_samples"],
        CFG["cfo_steps"],
        CFG["to_steps"],
    )

    for epoch in range(num_epochs):
        # ------------------------------------------------------------
        # 1. 학습 단계
        # ------------------------------------------------------------
        model.train()
        train_loss = 0.0

        for labels, snrs, cfos in train_loader:
            labels, snrs, cfos = labels.to(device), snrs.to(device), cfos.to(device)

            # 현재 배치에 대해 수신 파형을 시뮬레이터로 생성한다.
            rx_signals = simulator.generate_batch(labels, snrs, cfos, use_multipath=True)

            # 수신 파형을 다중 가설 2차원 특징맵으로 변환한다.
            features = simulator.extract_multi_hypothesis_bank(
                rx_signals,
                cfo_grid,
                to_grid,
                CFG["patch_size"],
            )

            # 일반적인 학습 루프이다.
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ------------------------------------------------------------
        # 2. 검증 단계
        # ------------------------------------------------------------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            # val_loader는 이미 고정된 features를 제공한다.
            for labels, features in val_loader:
                labels = labels.to(device)
                features = features.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total) * 100

        # validation loss가 가장 좋은 시점의 모델을 저장한다.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

    print(f"-> 검증 손실이 가장 낮았던 모델을 복원. (Best Val Loss: {best_val_loss:.4f})\n")

    model.load_state_dict(best_model_state)
    return model
