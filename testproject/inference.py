# -----------------------------------------------------------------------------
# inference.py – load trained model & output score 1‑10
# -----------------------------------------------------------------------------

import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from model import get_scoring_model

# Load the trained model
def load_scoring_model(weights_path=None, device=None):
    if weights_path is None:
        weights_path = Path(__file__).parent / "happyscore_resnet18.pt"
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = get_scoring_model().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device

# Predict the happiness score
def predict_happy_score(image_path_or_pil, model=None, device=None):
    if isinstance(image_path_or_pil, (str, Path)):
        pil = Image.open(image_path_or_pil)
    else:
        pil = image_path_or_pil

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    x = tf(pil).unsqueeze(0)

    if model is None:
        model, device = load_scoring_model()
    if device is None:
        device = next(model.parameters()).device

    x = x.to(device)
    with torch.no_grad():
        score01 = model(x).item()

    score = int(round(score01 * 9 + 1))  # Rescale 0-1 output to 1-10
    return score
