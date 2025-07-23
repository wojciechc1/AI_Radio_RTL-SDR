import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os

class FixedLengthAudioDataset(Dataset):
    def __init__(self, folder_path, segment_length=8192, noise_level=0.05):
        self.folder_path = folder_path
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])
        self.segment_length = segment_length
        self.noise_level = noise_level
        self.segments = self._split_all_files()

    def _split_all_files(self):
        segments = []
        for file in self.files:
            path = os.path.join(self.folder_path, file)
            waveform, sr = torchaudio.load(path)
            waveform = waveform.mean(dim=0, keepdim=True)  # mono

            total = waveform.shape[1]
            for i in range(0, total - self.segment_length + 1, self.segment_length):
                segment = waveform[:, i:i + self.segment_length]
                segments.append(segment)
        return segments

    def __len__(self):
        return len(self.segments)

    def generate_pink_noise(self, length=8192):
        # bialy szum
        white = torch.randn(length)

        # FFT do dziedziny częstotliwości
        f = torch.fft.rfft(white)

        # skala częstotliwości (1 / sqrt(f)) =/ 0 Hz
        freqs = torch.arange(1, f.shape[0] + 1, dtype=torch.float32)
        scale = 1 / torch.sqrt(freqs)

        # spektrum
        f_scaled = f * scale

        # dziedzina czasu
        pink = torch.fft.irfft(f_scaled, n=length)

        # normalizacja danych [-1;1]
        pink = pink / pink.abs().max()

        return pink.unsqueeze(0)  # [1, length]

    def add_radio_noise(self, clean, noise_level=0.05, impulse_prob=0.01):
        # różowy szum
        pink_noise = self.generate_pink_noise()
        noisy = clean + noise_level * pink_noise

        # impulsy
        impulses = torch.zeros_like(clean)
        for i in range(clean.shape[1]):
            if torch.rand(1).item() < impulse_prob:
                impulses[:, i] = torch.randn(1).item() * 5  # pik
        noisy = noisy + impulses

        # modulacja amplitudy
        modulation = (torch.sin(torch.linspace(0, 10 * 3.14, clean.shape[1])) + 1) / 2  # od 0 do 1
        noisy = noisy * modulation

        return noisy

    def __getitem__(self, idx):
        clean = self.segments[idx]
        #noise = self.noise_level * torch.randn_like(clean)
        #noisy = clean + noise
        noisy = self.add_radio_noise(clean)
        return noisy, clean
