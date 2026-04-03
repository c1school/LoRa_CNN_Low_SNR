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

def create_fixed_validation_set(simulator, num_samples: int, snr_range: tuple, max_cfo_bins: float):
    """
    다중경로 감쇠와 노이즈가 고정된 물리적 파형 텐서를 생성하여 반환하는 검증셋 생성기
    """
    device = simulator.device
    max_cfo_hz = max_cfo_bins * (simulator.bw / simulator.M)
    
    labels = torch.randint(0, simulator.M, (num_samples,), device=device)
    snrs = torch.empty(num_samples, device=device).uniform_(snr_range[0], snr_range[1])
    cfos = torch.empty(num_samples, device=device).uniform_(-max_cfo_hz, max_cfo_hz)
    
    rx_signals_list = []
    features_list = []
    batch_size = 2000
    
    print(">> 고정 검증 데이터셋(Fixed Validation Set) 생성 중...")
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            batch_rx = simulator.generate_batch(labels[i:end], snrs[i:end], cfos[i:end], use_multipath=True)
            batch_feat = simulator.extract_features(batch_rx)
            rx_signals_list.append(batch_rx.cpu())
            features_list.append(batch_feat.cpu())
            
    rx_signals = torch.cat(rx_signals_list)
    features = torch.cat(features_list)
    
    return TensorDataset(labels.cpu(), snrs.cpu(), cfos.cpu(), rx_signals, features)