import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from data_loader import train_loader

# Define the Happiness Model (ResNet50)
class HappinessModel(nn.Module):
    def __init__(self):
        super(HappinessModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 1)  # Regression output

    def forward(self, x):
        return self.model(x)

# Training Function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HappinessModel().to(device)

    criterion = nn.MSELoss()  # Regression loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Starting Training...")
    for epoch in range(10):
        for images, scores in train_loader:
            images, scores = images.to(device), scores.to(device).float()

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()

        print(f"✅ Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "happiness_model.pth")
    print("🎉 Training complete! Model saved.")

if __name__ == "__main__":
    train_model()