import torch
from torch.utils.data import Dataset

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

class FixedParametersDataset(Dataset):
    def __init__(self, M: int, num_samples: int, snr_range: tuple, max_cfo_bins: float, bw: float):
        self.labels = torch.randint(0, M, (num_samples,))
        self.snrs = torch.empty(num_samples).uniform_(snr_range[0], snr_range[1])
        max_cfo_hz = max_cfo_bins * (bw / M)
        self.cfos = torch.empty(num_samples).uniform_(-max_cfo_hz, max_cfo_hz)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx], self.snrs[idx], self.cfos[idx]