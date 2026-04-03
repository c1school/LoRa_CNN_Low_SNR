import torch
import torch.nn.functional as F
import numpy as np

class GPUOnlineSimulator:
    def __init__(self, sf: int = 7, bw: float = 125e3, fs: float = 1e6, device: str = "cuda"):
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.M = 2 ** sf
        self.Ts = self.M / bw
        self.N = int(self.Ts * fs)
        self.osr = self.N // self.M
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        t = torch.arange(self.N, device=self.device, dtype=torch.float32) / self.fs
        self.base_phase = torch.pi * (self.bw**2 / self.M) * (t**2)
        self.downchirp = torch.exp(-1j * self.base_phase)
        self.n_idx = torch.arange(self.N, device=self.device, dtype=torch.float32)

    def generate_batch(self, labels: torch.Tensor, snrs_db: torch.Tensor, cfos_hz: torch.Tensor, use_multipath: bool = True) -> torch.Tensor:
        batch_size = labels.size(0)
        
        tone_freq = (labels * self.osr).unsqueeze(1) * self.n_idx.unsqueeze(0) / self.N
        tone = torch.exp(1j * 2 * torch.pi * tone_freq)
        tx_signals = torch.exp(1j * self.base_phase.unsqueeze(0)) * tone

        if use_multipath:
            impaired_signals = tx_signals.clone()
            att1 = torch.empty(batch_size, 1, device=self.device).uniform_(0.3, 0.6)
            att2 = torch.empty(batch_size, 1, device=self.device).uniform_(0.1, 0.3)
            
            echo1 = torch.zeros_like(tx_signals)
            echo1[:, 3:] = tx_signals[:, :-3] * (att1 * 1j)
            
            echo2 = torch.zeros_like(tx_signals)
            echo2[:, 7:] = tx_signals[:, :-7] * att2
            
            impaired_signals = impaired_signals + echo1 + echo2
        else:
            impaired_signals = tx_signals

        t_matrix = self.n_idx.unsqueeze(0).repeat(batch_size, 1) / self.fs
        cfo_phase = 2 * torch.pi * cfos_hz.unsqueeze(1) * t_matrix
        impaired_signals = impaired_signals * torch.exp(1j * cfo_phase)

        signal_power = torch.mean(torch.abs(impaired_signals)**2, dim=1, keepdim=True)
        snr_linear = 10 ** (snrs_db.unsqueeze(1) / 10)
        noise_power = signal_power / snr_linear
        
        noise_real = torch.randn_like(impaired_signals, dtype=torch.float32)
        noise_imag = torch.randn_like(impaired_signals, dtype=torch.float32)
        noise = torch.sqrt(noise_power / 2) * (noise_real + 1j * noise_imag)

        return impaired_signals + noise

    def extract_features(self, rx_signals: torch.Tensor) -> torch.Tensor:
        dechirped = rx_signals * self.downchirp.unsqueeze(0)
        fft_complex = torch.fft.fft(dechirped, dim=1)
        
        max_vals = torch.max(torch.abs(fft_complex), dim=1, keepdim=True).values + 1e-10
        fft_norm = fft_complex / max_vals
        
        features = torch.stack((torch.real(fft_norm), torch.imag(fft_norm)), dim=1)
        return features

    def baseline_grouped_bin(self, rx_signals: torch.Tensor, window_size: int = 2):
        dechirped = rx_signals * self.downchirp.unsqueeze(0)
        fft_mag_sq = torch.abs(torch.fft.fft(dechirped, dim=1))**2
        batch_size = rx_signals.size(0)

        grouped_energy = torch.zeros(batch_size, self.M, device=self.device)
        for k in range(self.M):
            center = int(np.round(k * self.osr))
            indices = torch.arange(center - window_size, center + window_size + 1, device=self.device) % self.N
            grouped_energy[:, k] = torch.sum(fft_mag_sq[:, indices], dim=1)
            
        return grouped_energy, fft_mag_sq