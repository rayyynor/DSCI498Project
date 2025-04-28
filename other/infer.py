import torch
import cv2
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Define Model
class HappinessModel(nn.Module):
    def __init__(self):
        super(HappinessModel, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.model(x)

# Load Model
def load_model():
    model = HappinessModel()
    model.load_state_dict(torch.load("happiness_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_happiness(image_path):
    model = load_model()
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        score = model(image).item()

    return round(score, 2)

# Example Usage
if __name__ == "__main__":
    test_image = "test.jpg"
    print("Predicted Happiness Score:", predict_happiness(test_image))
