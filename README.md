## Running it yourself

**Train** — open `notebooks/mosquito_multihead_training.ipynb` in Colab, point `ROOT` at your unzipped dataset, run top to bottom.

**Deploy** — see [`mosquito_deploy/README.md`](mosquito_deploy/README.md) for the Cloud Run steps.

**Export for mobile** — `notebooks/mosquito_mobile_export.ipynb` handles ONNX → TFLite conversion, INT8 quantization, and latency benchmarking. This is the path that matters for VectorCam's actual phone-based deployment target.

**Probe the failure mode:**
```bash
python scripts/test_hard_cases.py \
  --root /path/to/mosquito_cdc \
  --api-url https://mosquito-classifier-207189004007.us-central1.run.app
```

## Stack

PyTorch · EfficientNet-B0 · FastAPI · Docker · Google Cloud Run · Comet ML · GitHub Pages

## Author

**Vishwanath Ninganolla** — [GitHub](https://github.com/523vishwanath) · [LinkedIn](https://linkedin.com/in/vishwanathninganolla)