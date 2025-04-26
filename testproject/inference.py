# -----------------------------------------------------------------------------
# inference.py – load trained model & output score 1‑10
# -----------------------------------------------------------------------------
import cv2
import torch                     # ← add this line
from PIL import Image
from pathlib import Path
from torchvision import transforms
from model import get_scoring_model

def load_scoring_model(weights_path="happyscore_resnet18.pt", device=None):
    if device is None:                       # ← choose automatically
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = get_scoring_model().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device

def predict_happy_score(image_path_or_pil, model=None, device=None):
    if isinstance(image_path_or_pil, (str, Path)):
        pil = Image.open(image_path_or_pil)
    else:
        pil = image_path_or_pil
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    x = tf(pil).unsqueeze(0)
    if model is None:
        model, device = load_scoring_model()
    x = x.to(device or next(model.parameters()).device)
    with torch.no_grad():
        score01 = model(x).item()
    score = int(round(score01 * 9 + 1))  # 1‑10
    return score
