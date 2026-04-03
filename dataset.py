import torch
from torch.utils.data import Dataset, TensorDataset

class OnlineParametersDataset(Dataset):
    def __init__(self, M: int, num_samples: int, snr_range: tuple, max_cfo_bins: float, bw: float):
        self.M = M
        self.num_samples = num_samples
        self.snr_range = snr_range
        self.max_cfo_hz = max_cfo_bins * (bw / M)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = torch.randint(0, self.M, (1,)).item()
        snr = torch.empty(1).uniform_(self.snr_range[0], self.snr_range[1]).item()
        cfo = torch.empty(1).uniform_(-self.max_cfo_hz, self.max_cfo_hz).item()
        return torch.tensor(label, dtype=torch.long), torch.tensor(snr, dtype=torch.float32), torch.tensor(cfo, dtype=torch.float32)

def create_fixed_feature_dataset(simulator, num_samples, snr_range, max_cfo_bins, seed=None):
    """학습 중 Validation용: 2D 다중 가설 Feature Bank 생성 및 저장"""
    if seed is not None: 
        torch.manual_seed(seed)
    device = simulator.device
    max_cfo_hz = max_cfo_bins * (simulator.bw / simulator.M)
    
    # 가설 격자 생성 (V7.0 핵심)
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(max_cfo_hz, max_to_samples=4, cfo_steps=17, to_steps=9)
    
    labels = torch.randint(0, simulator.M, (num_samples,), device=device)
    snrs = torch.empty(num_samples, device=device).uniform_(snr_range[0], snr_range[1])
    cfos = torch.empty(num_samples, device=device).uniform_(-max_cfo_hz, max_cfo_hz)
    
    features_list = []
    print("\n>> [V7.0] 2D Validation Feature Bank 생성 중... (메모리 및 시간 소요)")
    with torch.no_grad():
        # GPU 메모리 오버플로우 방지를 위해 배치 사이즈를 2000 -> 500으로 축소
        batch_size = 500
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            rx = simulator.generate_batch(labels[i:end], snrs[i:end], cfos[i:end], use_multipath=True)
            # 1D extract_features 대신 2D extract_multi_hypothesis_bank 사용
            feat = simulator.extract_multi_hypothesis_bank(rx, cfo_grid, to_grid)
            features_list.append(feat.cpu())
            
    return TensorDataset(labels.cpu(), torch.cat(features_list))

def create_fixed_waveform_dataset(simulator, num_samples_per_snr, snr_list, max_cfo_hz, seed=None):
    """Calibration 및 Test용: 파형 자체는 모델 구조와 무관하므로 V6와 동일하게 Rx Signal 유지"""
    if seed is not None: 
        torch.manual_seed(seed)
    device = simulator.device
    dataset_dict = {}
    
    for snr in snr_list:
        labels = torch.randint(0, simulator.M, (num_samples_per_snr,), device=device)
        snrs = torch.full((num_samples_per_snr,), snr, device=device)
        cfos = torch.empty(num_samples_per_snr, device=device).uniform_(-max_cfo_hz, max_cfo_hz)
        
        rx_list = []
        with torch.no_grad():
            for i in range(0, num_samples_per_snr, 2000):
                end = min(i + 2000, num_samples_per_snr)
                rx = simulator.generate_batch(labels[i:end], snrs[i:end], cfos[i:end], use_multipath=True)
                rx_list.append(rx.cpu())
        
        dataset_dict[snr] = TensorDataset(labels.cpu(), torch.cat(rx_list))
    return dataset_dict