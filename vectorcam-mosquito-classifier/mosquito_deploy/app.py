"""
Mosquito species + sex classifier — inference API
Deploy on Hugging Face Spaces (Docker SDK) or any server that can run FastAPI.
"""

import io
import json
import os

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision.models import efficientnet_b0

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# Model definition — MUST match the architecture used in training exactly
# ---------------------------------------------------------------------------
class MosquitoMultiHead(nn.Module):
    def __init__(self, n_species, n_sex):
        super().__init__()
        backbone = efficientnet_b0(weights=None)  # weights loaded from checkpoint below
        self.trunk = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.classifier[1].in_features  # 1280

        self.species_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, n_species))
        self.sex_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, n_sex))

    def forward(self, x):
        feats = self.pool(self.trunk(x)).flatten(1)
        return {"species": self.species_head(feats), "sex": self.sex_head(feats)}


# ---------------------------------------------------------------------------
# Load checkpoint at startup
# ---------------------------------------------------------------------------
CKPT_PATH = os.environ.get("MODEL_CKPT_PATH", "mosquito_multihead_effnetb0_final.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT_PATH, map_location=device)

species_map = ckpt["species_map"]          # name -> id
sex_map = ckpt["sex_map"]                  # name -> id
species_names = {v: k for k, v in species_map.items()}  # id -> name
sex_names = {v: k for k, v in sex_map.items()}
IMG_SIZE = ckpt.get("img_size", 224)

model = MosquitoMultiHead(n_species=len(species_map), n_sex=len(sex_map)).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ---------------------------------------------------------------------------
# Preprocessing — must match eval_tfms used in training (no augmentation, just resize+normalize)
# ---------------------------------------------------------------------------
eval_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def tta_predict(img_tensor):
    """Average predictions over original + horizontal flip + vertical flip."""
    variants = [img_tensor, torch.flip(img_tensor, dims=[3]), torch.flip(img_tensor, dims=[2])]
    species_probs, sex_probs = 0, 0
    with torch.no_grad():
        for v in variants:
            preds = model(v)
            species_probs = species_probs + torch.softmax(preds["species"], dim=1)
            sex_probs = sex_probs + torch.softmax(preds["sex"], dim=1)
    return species_probs / len(variants), sex_probs / len(variants)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Mosquito Species + Sex Classifier")

# Allow calls from a browser-based frontend if you build one later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "species_classes": list(species_map.keys()),
        "sex_classes": list(sex_map.keys()),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)

    tensor = eval_tfms(image=img_np)["image"].unsqueeze(0).to(device)

    species_probs, sex_probs = tta_predict(tensor)

    species_id = species_probs.argmax(1).item()
    sex_id = sex_probs.argmax(1).item()

    return {
        "species": species_names[species_id],
        "species_confidence": round(species_probs[0, species_id].item(), 4),
        "sex": sex_names[sex_id],
        "sex_confidence": round(sex_probs[0, sex_id].item(), 4),
        "species_all_probs": {
            species_names[i]: round(p, 4) for i, p in enumerate(species_probs[0].tolist())
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
