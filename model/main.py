
from dataset import FixedLengthAudioDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from net import UNet1D
from train import train
from eval import evaluate as eval

train_dataset = FixedLengthAudioDataset('./data/train')
eval_dataset = FixedLengthAudioDataset('./data/eval')

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=20, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet1D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 4


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")


    train(model, device, optimizer, criterion, train_loader)
    eval(model, device, criterion, eval_loader)


torch.save(model.state_dict(), "trained_model_1.pth")

