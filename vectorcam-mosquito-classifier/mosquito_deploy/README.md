---
title: Mosquito Species Sex Classifier
emoji: 🦟
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Mosquito Species + Sex Classifier

FastAPI inference service for a two-head EfficientNet-B0 classifier
(species: 7 classes, sex: 2 classes), trained on CDC/MR4 mosquito images
as a portfolio project supporting a VectorCam-style pipeline.

## Endpoints

- `GET /` — health check, lists the class names the model supports
- `POST /predict` — upload an image, get back predicted species + sex with confidence scores

## Local test

```bash
docker build -t mosquito-classifier .
docker run -p 7860:7860 mosquito-classifier
curl -X POST -F "file=@test.jpg" http://localhost:7860/predict
```

## Files

- `app.py` — FastAPI app + model definition + inference logic
- `mosquito_multihead_effnetb0_final.pt` — trained checkpoint (add this yourself, see below)
- `requirements.txt` — pinned Python deps
- `Dockerfile` — container build for Hugging Face Spaces Docker SDK

## Adding your trained model

This repo does not include the trained weights. Download
`mosquito_multihead_effnetb0_final.pt` from your Colab training run
(Section 13 of the training notebook — it auto-downloads via
`files.download(...)`), then drop it into this folder before building
or pushing.
