import torch
import torch.nn as nn


class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()

        # encoder (w dół)
        self.enc1 = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(16))
        self.enc2 = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32))
        self.enc3 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64))

        # decoder (w górę)
        self.dec2 = nn.Sequential(nn.Conv1d(64 + 32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32))
        self.dec1 = nn.Sequential(nn.Conv1d(32 + 16, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(16))

        # output
        self.final = nn.Conv1d(16, 1, 1)

        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        # decoder
        d2 = self.up(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out
