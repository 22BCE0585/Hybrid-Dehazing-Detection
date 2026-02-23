import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

from models.cnn_refiner.model import TransmissionRefiner
from models.cnn_refiner.dataset import ResidualTransmissionDataset


def train_model(root_dir, epochs=20, lr=1e-4, batch_size=8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = ResidualTransmissionDataset(root_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    model = TransmissionRefiner().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        model.train()
        train_loss = 0

        for hazy, trans_gt in tqdm(train_loader):
            hazy = hazy.to(device)
            trans_gt = trans_gt.to(device)

            pred = model(hazy)

            loss = criterion(pred, trans_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for hazy, trans_gt in val_loader:
                hazy = hazy.to(device)
                trans_gt = trans_gt.to(device)

                pred = model(hazy)
                loss = criterion(pred, trans_gt)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print("-" * 40)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models/cnn_refiner/checkpoints", exist_ok=True)
            torch.save(model.state_dict(),
                    "models/cnn_refiner/checkpoints/best_model.pth")
            print("Model saved.")

    print("Training complete.")