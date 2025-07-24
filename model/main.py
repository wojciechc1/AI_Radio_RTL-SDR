
from dataset import FixedLengthAudioDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from net import UNet1D
from train import train
from eval import evaluate as eval


import torch

class HybridLoss:
    def __init__(self, alpha=1.0, beta=1.0, n_fft=512, hop_length=128):
        self.alpha = alpha
        self.beta = beta
        self.n_fft = n_fft
        self.hop_length = hop_length

    def stft_loss(self, clean, denoised):
        spec_clean = torch.stft(clean.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        spec_denoised = torch.stft(denoised.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        return F.l1_loss(torch.abs(spec_clean), torch.abs(spec_denoised))

    def __call__(self, clean, denoised):
        l1 = F.l1_loss(denoised, clean)
        stft = self.stft_loss(clean, denoised)
        return self.alpha * l1 + self.beta * stft



# data
train_dataset = FixedLengthAudioDataset('./data/train')
eval_dataset = FixedLengthAudioDataset('./data/eval')

# data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# model params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet1D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = HybridLoss()
num_epochs = 4


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")


    train(model, device, optimizer, criterion, train_loader)
    eval(model, device, criterion, eval_loader)


torch.save(model.state_dict(), "saved_models/trained_model_1.pth")

