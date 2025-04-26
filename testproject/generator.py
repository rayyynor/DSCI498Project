# -----------------------------------------------------------------------------
# generator.py – Stable‑Diffusion‑based happy‑face editing
# -----------------------------------------------------------------------------

import torch

SD_MODEL = "runwayml/stable-diffusion-v1-5"  # good generic SD‑v1‑5
_HAPPY_PROMPT = (
    "a high‑resolution photograph of the same person, joyful big smile, cheerful, happy, natural lighting"
)
_NEGATIVE = (
    "deformed, ugly, bad anatomy, disfigured, cartoon, out of frame, lowres, kappa"
)

def init_img2img(device="cuda"):
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL,
        safety_checker=None,
    ).to(device)
    
    
    return pipe


@torch.inference_mode()
def make_happier_face(pil_image, strength=0.6, guidance_scale=8.0, seed=42):
    """Return a PIL.Image with noticeably happier expression."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = init_img2img(device)
    generator = torch.Generator(device).manual_seed(seed)
    out = pipe(
        prompt=_HAPPY_PROMPT,
        image=pil_image,
        negative_prompt=_NEGATIVE,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return out.images[0]
