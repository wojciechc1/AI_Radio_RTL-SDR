

def train(model, device, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0

    for noisy, clean in dataloader:

        noisy = noisy.to(device)  # X
        clean = clean.to(device)  # Y

        optimizer.zero_grad()

        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"train loss: {avg_loss}")

