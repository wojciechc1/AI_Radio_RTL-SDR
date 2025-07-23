import torch


def evaluate(model, device, criterion, dataloader):
    model.eval()
    total_loss = 0


    with torch.no_grad():
        for noisy, clean in dataloader:

            noisy = noisy.to(device)  # X
            clean = clean.to(device)  # Y


            output = model(noisy)
            loss = criterion(output, clean)


            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"eval loss: {avg_loss:.4f}")

