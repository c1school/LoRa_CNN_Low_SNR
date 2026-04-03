import torch
import torch.nn as nn
import torch.optim as optim

def train_online_model(model, simulator, train_loader, val_loader, max_cfo_hz, num_epochs=20, lr=0.0005):
    device = simulator.device
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n[V7.0 2D-CNN 온라인 학습 시작] Device: {device}, Epochs: {num_epochs}")
    best_val_loss = float("inf")
    best_model_state = None

    # 훈련 내내 사용할 가설 격자(Grid) 미리 생성
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(max_cfo_hz, max_to_samples=4, cfo_steps=17, to_steps=9)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for labels, snrs, cfos in train_loader:
            labels, snrs, cfos = labels.to(device), snrs.to(device), cfos.to(device)
            
            # 파형 생성
            rx_signals = simulator.generate_batch(labels, snrs, cfos, use_multipath=True)
            
            # [V7.0] 2D 다중 가설 증거 볼륨 추출 (Shape: [Batch, 2, 153, 128])
            features = simulator.extract_multi_hypothesis_bank(rx_signals, cfo_grid, to_grid)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for labels, features in val_loader:
                labels = labels.to(device)
                features = features.to(device) # Validation 셋은 이미 2D로 저장되어 있음
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total) * 100

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print(f"-> 최적 모델 복원 완료 (Best Val Loss: {best_val_loss:.4f})\n")
    model.load_state_dict(best_model_state)
    return model