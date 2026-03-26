# ============================================================
#모듈화 이전 main.py입니다.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import time
import os
from scipy.signal import lfilter

# ============================================================
# 0. 실험 재현성 확보
# ------------------------------------------------------------
# 같은 코드와 같은 설정으로 다시 실행했을 때 결과가 크게 달라지지 않도록
# random seed를 고정하는 부분임.
#
# - numpy seed 고정
# - torch seed 고정
# - CUDA 사용 시 GPU seed도 고정
# - PYTHONHASHSEED 고정
# - cudnn deterministic 모드 활성화
#
# ============================================================
def set_seed(seed=2026):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ============================================================
# 1. Baseline 시뮬레이터
# ------------------------------------------------------------
# 이 클래스는 LoRa-like 심볼을 생성하고,
# 채널 왜곡(CFO, Multipath, AWGN)을 적용하고,
# dechirp + FFT 기반 특징을 뽑고,
# classical baseline 복조까지 수행하는 핵심 시뮬레이터임.
#
# "송신기 + 채널 + 수신기 전처리" 역할을 동시에 담당한다고 보면 됨.
# ============================================================
class LoRaResearchSimulator:
    def __init__(self, sf=7, bw=125e3, fs=1e6):
        # SF: spreading factor
        # BW: bandwidth
        # FS: sampling frequency
        self.sf = sf
        self.bw = bw
        self.fs = fs

        # LoRa 심볼 개수 M = 2^SF
        self.M = 2**sf

        # 심볼 길이 Ts = M / BW
        self.Ts = self.M / bw

        # 한 심볼을 샘플링한 총 샘플 수
        self.N = int(self.Ts * fs)

        # Oversampling ratio
        # 현재 설정에서는 FFT peak가 symbol 그 자체가 아니라
        # symbol * osr 위치 부근에 나타나므로 baseline에서 이 값이 중요함.
        self.osr = self.N // self.M

        # --------------------------------------------------------
        # base chirp 생성
        # --------------------------------------------------------
        # 이산시간 chirp의 위상(base_phase)을 정의함.
        # 이후 generate_symbol()에서는 이 위상에 tone 성분을 얹어
        # 각 심볼에 해당하는 upchirp를 생성함.
        t = np.arange(self.N) / self.fs
        self.base_phase = np.pi * (self.bw**2 / self.M) * (t**2)

        # dechirp에 사용할 기준 downchirp
        # 수신 신호에 이 값을 곱하면 원래 chirp 구조를 제거하고
        # 심볼에 대응하는 tone 성분만 남기게 됨.
        self.downchirp = np.exp(-1j * self.base_phase)

    def generate_symbol(self, symbol):
        # --------------------------------------------------------
        # 특정 symbol index에 대응되는 송신 심볼 생성
        # --------------------------------------------------------
        # 핵심 아이디어:
        #   기본 chirp 위에 symbol * osr 에 대응되는 tone을 얹어
        #   dechirp 후 FFT peak가 정해진 위치에 나타나게 함.
        #
        # 즉, 이 함수는 "심볼 번호 -> 실제 복소 baseband 파형" 변환기 역할을 함.
        n = np.arange(self.N)
        tone = np.exp(1j * 2 * np.pi * (symbol * self.osr) * n / self.N)
        return np.exp(1j * self.base_phase) * tone

    def apply_impaired_channel(self, signal, snr_db, impairment_config):
        # --------------------------------------------------------
        # 채널 왜곡 적용 함수
        # --------------------------------------------------------
        # 입력 clean signal에 다음 왜곡을 순서대로 적용함.
        #
        # 1) Multipath
        # 2) CFO
        # 3) AWGN
        #
        # impairment_config에 따라 어떤 왜곡을 켤지/끄는지 결정함.
        impaired_signal = np.copy(signal)

        # -------------------------
        # Multipath 적용
        # -------------------------
        # 다중경로를 tapped delay line 형태로 모델링함.
        # 예: taps=[1.0, 0.4j, 0.2], delays=[0,2,5]
        # 이는 직진파 + 지연된 반사파들이 함께 들어오는 상황을 모사함.
        if impairment_config.get('use_multipath', False):
            taps = impairment_config.get('multipath_taps', [1.0, 0.4j, 0.2])
            delays = impairment_config.get('multipath_delays', [0, 2, 5])
            h = np.zeros(max(delays) + 1, dtype=np.complex128)
            h[delays] = taps
            impaired_signal = lfilter(h, 1, impaired_signal)

        # -------------------------
        # CFO 적용
        # -------------------------
        # CFO(Carrier Frequency Offset)는 송수신기 주파수 미스매치를 나타냄.
        # bin_spacing = BW / M 을 기준으로 fractional-bin 수준의 주파수 오프셋을 부여함.
        if impairment_config.get('use_cfo', False):
            bin_spacing = self.bw / self.M
            max_cfo_bins = impairment_config.get('max_cfo_bins', 0.35)
            cfo_hz = np.random.uniform(-max_cfo_bins, max_cfo_bins) * bin_spacing
            cfo_phase = 2 * np.pi * cfo_hz * (np.arange(self.N) / self.fs)
            impaired_signal = impaired_signal * np.exp(1j * cfo_phase)

        # -------------------------
        # AWGN 적용
        # -------------------------
        # 목표 SNR에 맞는 복소 Gaussian noise를 추가함.
        signal_power = np.mean(np.abs(impaired_signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.N) + 1j * np.random.randn(self.N)
        )

        return impaired_signal + noise

    # --------------------------------------------------------
    # Complex 입력 특징 추출기
    # --------------------------------------------------------
    # dechirp 후 FFT를 하고, 그 복소수 결과를
    # [Real, Imag] 2채널 입력으로 반환함.
    #
    # 즉, amplitude뿐 아니라 phase 정보도 보존하는 특징 표현임.
    # 내가 하려는고자 하는거임.
    def dechirp_and_fft_complex(self, iq_signal):
        dechirped = iq_signal * self.downchirp
        fft_complex = np.fft.fft(dechirped)

        # 전체 크기를 0~1 정도로 맞추기 위한 정규화
        max_val = np.max(np.abs(fft_complex)) + 1e-10
        fft_norm = fft_complex / max_val

        # shape = (2, N)
        # 0번 채널: Real part
        # 1번 채널: Imag part
        return np.vstack((np.real(fft_norm), np.imag(fft_norm)))

    # --------------------------------------------------------
    # Magnitude-only 입력 특징 추출기
    # --------------------------------------------------------
    # dechirp 후 FFT magnitude만 남기고 위상은 버림.
    # ablation study에서 "위상 정보가 정말 도움이 되는가?"를 보기 위한 비교군임.
    def dechirp_and_fft_mag(self, iq_signal):
        dechirped = iq_signal * self.downchirp
        fft_mag = np.abs(np.fft.fft(dechirped))
        max_val = np.max(fft_mag) + 1e-10
        fft_norm = fft_mag / max_val

        # shape = (1, N)
        return np.expand_dims(fft_norm, axis=0)

    # --------------------------------------------------------
    # Naive baseline 복조기
    # --------------------------------------------------------
    # 가장 큰 FFT peak 하나만 찾고,
    # 그 위치를 osr로 나누어 symbol로 환산하는 고전적 복조 방식임.
    #
    # classical detector 중 가장 단순한 형태라고 볼 수 있음.
    def baseline_demod_naive(self, rx_signal):
        dechirped = rx_signal * self.downchirp
        fft_mag = np.abs(np.fft.fft(dechirped))
        peak_idx = int(np.argmax(fft_mag))
        return int(np.round(peak_idx / self.osr)) % self.M

    # --------------------------------------------------------
    # Grouped-bin classical baseline
    # --------------------------------------------------------
    # CFO나 spectral leakage가 존재할 때 peak가 한 bin에만 생기지 않고
    # 주변 bin으로 퍼질 수 있으므로, 중심 bin 주변 에너지를 합산하여 판정함.
    #
    # naive baseline보다 조금 더 robust한 classical detector 역할을 함.
    def baseline_demod_grouped_bin(self, rx_signal, window_size=2):
        dechirped = rx_signal * self.downchirp
        fft_mag_sq = np.abs(np.fft.fft(dechirped))**2

        grouped_energy = np.zeros(self.M)
        for k in range(self.M):
            center = int(np.round(k * self.osr))
            indices = np.mod(np.arange(center - window_size, center + window_size + 1), self.N)
            grouped_energy[k] = np.sum(fft_mag_sq[indices])

        return int(np.argmax(grouped_energy))

# ============================================================
# 2. Dataset 클래스
# ------------------------------------------------------------
# feature_type에 따라
#   - complex 입력 (2채널)
#   - mag 입력 (1채널)
# 을 생성할 수 있도록 만든 데이터셋임.
#
# 즉, 같은 채널 환경에서 "입력 표현"만 다르게 하여
# complex CNN vs magnitude-only CNN을 공정하게 비교하기 위한 용도임.
# ============================================================
class LoRaResearchDataset(Dataset):
    def __init__(self, simulator, num_samples, snr_range, impairment_config, mode='train', feature_type='complex'):
        self.simulator = simulator
        self.num_samples = num_samples
        self.feature_type = feature_type

        print(f"[{mode.upper()} | {feature_type.upper()}] {num_samples}개의 데이터 생성 중...")
        self.data_x = []
        self.data_y = []

        for i in range(num_samples):
            # eval 모드에서는 내부 seed를 고정해 평가 표본을 재현 가능하게 함
            if mode == 'eval':
                np.random.seed(2026 + i)

            # 랜덤 label 선택
            label = np.random.randint(0, self.simulator.M)

            # clean signal 생성
            clean_sig = self.simulator.generate_symbol(label)

            # snr_range가 tuple이면 구간 내 랜덤 SNR
            # 아니면 고정 SNR
            target_snr = np.random.uniform(snr_range[0], snr_range[1]) if isinstance(snr_range, tuple) else snr_range

            # 채널 왜곡 적용
            noisy_sig = self.simulator.apply_impaired_channel(clean_sig, target_snr, impairment_config)

            # feature_type에 따라 특징 추출 방식 선택
            if self.feature_type == 'complex':
                features = self.simulator.dechirp_and_fft_complex(noisy_sig)
            else:
                features = self.simulator.dechirp_and_fft_mag(noisy_sig)

            self.data_x.append(torch.tensor(features, dtype=torch.float32))
            self.data_y.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

# ============================================================
# 3. 1D CNN 범용 아키텍처
# ------------------------------------------------------------
# complex 입력(2채널)과 mag 입력(1채널)을 같은 구조 위에서 비교하기 위해
# in_channels만 바꿀 수 있도록 만든 범용 CNN임.
#
# 구조:
# Conv-BN-ReLU-Pool 블록 여러 개 -> Flatten -> FC -> Dropout -> FC
# ============================================================
class LoRaCNN(nn.Module):
    def __init__(self, num_classes, input_length, in_channels=2):
        super(LoRaCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        # FC 입력 차원을 자동 계산
        with torch.no_grad():
            dummy_x = torch.zeros(1, in_channels, input_length)
            dummy_x = self.pool1(F.relu(self.bn1(self.conv1(dummy_x))))
            dummy_x = self.pool2(F.relu(self.bn2(self.conv2(dummy_x))))
            dummy_x = self.pool3(F.relu(self.bn3(self.conv3(dummy_x))))
            dummy_x = F.relu(self.bn4(self.conv4(dummy_x)))
            flattened_size = dummy_x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================
# 4. 학습 루프
# ------------------------------------------------------------
# Train / Validation을 분리해서 수행하고,
# Validation Loss가 가장 낮은 시점의 모델을 best checkpoint로 저장/복원함.
#
# 즉, 마지막 epoch 모델이 아니라 "가장 일반화가 잘된 모델"을 최종 모델로 선택함.
# ============================================================
def train_research_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"\n[학습 시작] Device: {device}, Epochs: {num_epochs}")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # -------------------------
        # Training phase
        # -------------------------
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # -------------------------
        # Validation phase
        # -------------------------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total) * 100

        # Validation Loss가 더 낮으면 best checkpoint 갱신
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 진행 상황 출력
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 가장 좋은 validation 성능을 보인 시점의 모델 복원
    print(f"-> 가장 낮은 Val Loss({best_val_loss:.4f}) 모델을 복원합니다.\n")
    model.load_state_dict(best_model_state)
    return model

# ============================================================
# 5. Threshold Gain 계산 함수
# ------------------------------------------------------------
# 특정 SER(예: 1e-1, 1e-2)를 달성하는 데 필요한 SNR을
# 선형 보간으로 계산하는 함수임.
#
# 그래프를 눈으로만 보는 것이 아니라,
# "baseline 대비 CNN이 몇 dB 개선되었는가"를 수치화하기 위한 도구임.
# ============================================================
def calculate_snr_for_target_ser(snrs, sers, target_ser):
    """선형 보간을 통해 특정 SER 달성에 필요한 SNR을 계산"""
    for i in range(len(sers)-1):
        if sers[i] >= target_ser >= sers[i+1]:
            slope = (snrs[i+1] - snrs[i]) / (sers[i+1] - sers[i] + 1e-12)
            return snrs[i] + slope * (target_ser - sers[i])
    return None

# ============================================================
# 6. Ablation + Benchmark 평가 함수
# ------------------------------------------------------------
# 이 함수는 같은 noisy sample에 대해 아래 네 방법을 동시에 비교함.
#
# 1) Naive FFT baseline
# 2) Grouped-bin classical baseline
# 3) Magnitude-only CNN
# 4) Complex-input CNN
#
# 즉, 단순 성능 비교가 아니라
# - baseline 대비 CNN 이득
# - magnitude-only 대비 complex-input 이득
# - seen / unseen 채널에서 일반화 한계
# 를 모두 동시에 확인할 수 있는 핵심 평가 함수임.
# ============================================================
def evaluate_ablation_model(model_complex, model_mag, simulator, snr_list, impairment_config, benchmark_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_complex.to(device).eval()
    model_mag.to(device).eval()

    # 각 방법별 SER 곡선 저장용
    results = {'Complex CNN': [], 'Mag CNN': [], 'Grouped': [], 'Naive': []}

    print(f"\n[정밀 평가 시작 : {benchmark_name}]")
    print("-" * 80)

    for snr in snr_list:
        # --------------------------------------------------------
        # Monte Carlo 샘플 수 동적 조정
        # --------------------------------------------------------
        # 저오류율 구간일수록 더 많은 샘플을 써야 SER가 통계적으로 의미 있음.
        if snr >= -10:
            num_test_samples = 50000
        elif snr >= -15:
            num_test_samples = 10000
        else:
            num_test_samples = 3000

        print(f"SNR {snr:3d} dB 평가 중... (Monte Carlo: {num_test_samples})")

        # 정답 개수 카운터
        cor = {'Complex CNN': 0, 'Mag CNN': 0, 'Grouped': 0, 'Naive': 0}

        batch_size = 1000
        num_batches = num_test_samples // batch_size

        with torch.no_grad():
            for _ in range(num_batches):
                labels_np = np.random.randint(0, simulator.M, batch_size)
                features_c, features_m = [], []

                for i, lbl in enumerate(labels_np):
                    clean_sig = simulator.generate_symbol(lbl)
                    noisy_sig = simulator.apply_impaired_channel(clean_sig, snr, impairment_config)

                    # -------------------------
                    # baseline 평가
                    # -------------------------
                    if simulator.baseline_demod_naive(noisy_sig) == lbl:
                        cor['Naive'] += 1
                    if simulator.baseline_demod_grouped_bin(noisy_sig, window_size=2) == lbl:
                        cor['Grouped'] += 1

                    # -------------------------
                    # CNN 입력 특징 추출
                    # -------------------------
                    features_c.append(torch.tensor(simulator.dechirp_and_fft_complex(noisy_sig), dtype=torch.float32))
                    features_m.append(torch.tensor(simulator.dechirp_and_fft_mag(noisy_sig), dtype=torch.float32))

                # -------------------------
                # CNN 평가
                # -------------------------
                out_c = model_complex(torch.stack(features_c).to(device))
                out_m = model_mag(torch.stack(features_m).to(device))
                lbl_t = torch.tensor(labels_np, dtype=torch.long).to(device)

                cor['Complex CNN'] += (torch.max(out_c.data, 1)[1] == lbl_t).sum().item()
                cor['Mag CNN'] += (torch.max(out_m.data, 1)[1] == lbl_t).sum().item()

        # SER 계산
        for key in results.keys():
            results[key].append(1.0 - (cor[key] / num_test_samples))

        print(f"  -> SER [Comp CNN]:{results['Complex CNN'][-1]:.5f} | [Mag CNN]:{results['Mag CNN'][-1]:.5f} | [Grp]:{results['Grouped'][-1]:.5f}")

    # ========================================================
    # 그래프 저장
    # ========================================================
    filename = f"ser_curve_{benchmark_name.lower().replace(' ', '_')}.png"
    plt.figure(figsize=(10, 7))
    plt.semilogy(snr_list, results['Naive'], marker='x', linestyle=':', color='gray', alpha=0.5, label='Naive FFT')
    #plt.semilogy(snr_list, results['Grouped'], marker='s', linestyle='--', color='black', label='Grouped-Bin Classical')
    #plt.semilogy(snr_list, results['Mag CNN'], marker='^', linestyle='-.', color='orange', label='Mag Input CNN (Ablation)')
    plt.semilogy(snr_list, results['Complex CNN'], marker='o', linestyle='-', color='red', linewidth=2, label='Complex Input CNN')

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=12)
    plt.ylabel('Symbol Error Rate (SER)', fontsize=12)
    plt.title(f'LoRa Demodulation Performance: {benchmark_name}', fontsize=14)
    plt.ylim([5e-5, 1.1])
    plt.legend(fontsize=11, loc='lower left')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================================
    # Threshold Gain 정량화 표 출력
    # --------------------------------------------------------
    # Grouped baseline, Mag CNN, Complex CNN에 대해
    # SER=1e-1 / 1e-2를 달성하는 SNR을 자동 계산해 표로 출력함.
    # ========================================================
    print(f"\n[{benchmark_name}] Threshold Gain 분석표")
    print("=" * 60)
    print(f"{'Method':<20} | {'SNR for SER=1e-1':<15} | {'SNR for SER=1e-2':<15}")
    print("-" * 60)
    for key in ['Grouped', 'Mag CNN', 'Complex CNN']:
        snr_1e1 = calculate_snr_for_target_ser(snrs=snr_list, sers=results[key], target_ser=1e-1)
        snr_1e2 = calculate_snr_for_target_ser(snrs=snr_list, sers=results[key], target_ser=1e-2)

        # None 여부로 판정하는 것이 안전함
        str_1e1 = f"{snr_1e1:.2f} dB" if snr_1e1 is not None else "N/A"
        str_1e2 = f"{snr_1e2:.2f} dB" if snr_1e2 is not None else "N/A"

        print(f"{key:<20} | {str_1e1:<15} | {str_1e2:<15}")
    print("=" * 60)

    return results

# ============================================================
# 7. 메인 실행 블록
# ------------------------------------------------------------
#
# 순서:
# 1) 시뮬레이터 생성
# 2) Complex CNN 초기화/학습(or 로드)
# 3) Mag CNN 초기화/학습(or 로드)
# 4) 3개 시나리오에서 benchmark 수행
#    - Scenario A: Pure AWGN
#    - Scenario B: Seen Impaired
#    - Scenario C: Unseen Impaired
# ============================================================
if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)

    # LoRa-like simulator 생성
    sim = LoRaResearchSimulator(sf=7, bw=125e3, fs=1e6)

    # --------------------------------------------------------
    # 두 모델 초기화
    # --------------------------------------------------------
    # Complex CNN: Real/Imag 2채널 입력
    # Mag CNN: magnitude-only 1채널 입력
    model_comp = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=2)
    model_mag = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=1)

    path_comp = 'saved_models/lora_comp_cnn_v2.pth'
    path_mag = 'saved_models/lora_mag_cnn_v2.pth'

    # 학습 시 사용할 impaired 환경
    train_config = {'use_cfo': True, 'max_cfo_bins': 0.35, 'use_multipath': True}
    total_samples = 40000

    # --------------------------------------------------------
    # 1. Complex CNN 훈련 또는 로드
    # --------------------------------------------------------
    if os.path.exists(path_comp):
        # 이미 학습된 가중치가 있으면 재사용
        model_comp.load_state_dict(torch.load(path_comp, map_location=torch.device('cpu')))
    else:
        print(">> [학습 1/2] Complex CNN 훈련 중...")
        ds_comp = LoRaResearchDataset(
            sim, total_samples, (-20, 0), train_config, feature_type='complex'
        )
        dl_train, dl_val = random_split(
            ds_comp,
            [int(0.85*total_samples), total_samples - int(0.85*total_samples)]
        )
        model_comp = train_research_model(
            model_comp,
            DataLoader(dl_train, batch_size=256, shuffle=True),
            DataLoader(dl_val, batch_size=256),
            num_epochs=25
        )
        torch.save(model_comp.state_dict(), path_comp)

    # --------------------------------------------------------
    # 2. Mag CNN 훈련 또는 로드
    # --------------------------------------------------------
    if os.path.exists(path_mag):
        model_mag.load_state_dict(torch.load(path_mag, map_location=torch.device('cpu')))
    else:
        print("\n>> [학습 2/2] Mag CNN (Ablation) 훈련 중...")
        ds_mag = LoRaResearchDataset(
            sim, total_samples, (-20, 0), train_config, feature_type='mag'
        )
        dl_train, dl_val = random_split(
            ds_mag,
            [int(0.85*total_samples), total_samples - int(0.85*total_samples)]
        )
        model_mag = train_research_model(
            model_mag,
            DataLoader(dl_train, batch_size=256, shuffle=True),
            DataLoader(dl_val, batch_size=256),
            num_epochs=25
        )
        torch.save(model_mag.state_dict(), path_mag)

    # 평가할 SNR 점들
    test_snrs = np.arange(-25, 1, 2)

    # --------------------------------------------------------
    # Scenario A: Pure AWGN
    # --------------------------------------------------------
    # classical detector가 잘 작동하는 기본 환경
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        {'use_cfo': False, 'use_multipath': False},
        "Scenario A - Pure AWGN"
    )

    # --------------------------------------------------------
    # Scenario B: Seen Impaired
    # --------------------------------------------------------
    # 학습 시 사용했던 것과 동일한 impaired 채널
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        train_config,
        "Scenario B - Seen Impaired"
    )

    # --------------------------------------------------------
    # Scenario C: Unseen Impaired
    # --------------------------------------------------------
    # 학습 시 보지 못한 더 안좋은 환경의 채널
    # 일반화 성능과 robustness 한계를 보기 위한 시나리오
    config_unseen = {
        'use_cfo': True,
        'max_cfo_bins': 0.45,          # CFO 범위 증가
        'use_multipath': True,
        'multipath_taps': [0.9, -0.6j, 0.4],  # 학습 때와 다른 tap 패턴
        'multipath_delays': [0, 4, 9]         # 더 긴 delay
    }
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        config_unseen,
        "Scenario C - Unseen Impaired"
    )

    print("\n========== [완료] ==========")