# -----------------------------------------------------------------------------
# train.py – fine‑tune on FER2013
# -----------------------------------------------------------------------------
from tqdm import tqdm
from data_utils import download_fer2013, FERDataset, CSV_FILE
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import get_scoring_model

def train_scoring_model(batch_size=128, epochs=10, lr=1e-3, device=None):
    download_fer2013()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tf_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    tf_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
    ])

    train_ds = FERDataset(CSV_FILE, "Training", tf_train)
    val_ds = FERDataset(CSV_FILE, "PublicTest", tf_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_scoring_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), (y / 10.0).unsqueeze(1).to(device)  # scale 0‑1
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        print(f"Train loss: {epoch_loss/len(train_ds):.4f}")
        # quick val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), (y / 10.0).unsqueeze(1).to(device)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
        print(f"Val loss: {val_loss/len(val_ds):.4f}")
    torch.save(model.state_dict(), "happyscore_resnet18.pt")
    print("[train] model saved to happyscore_resnet18.pt")
