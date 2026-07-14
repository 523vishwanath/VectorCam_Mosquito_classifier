## Running it yourself

**Train** — open `notebooks/mosquito_multihead_training.ipynb` in Colab (A100 recommended), set `ROOT` to your unzipped dataset, run top to bottom. It builds `labels.csv`, trains all three stages, logs to Comet, and writes the checkpoint.

**Deploy** — see [`mosquito_deploy/README.md`](mosquito_deploy/README.md) for the full Cloud Run walkthrough. In short: `gcloud run deploy --source .` builds the Dockerfile and returns a public URL.

**Export for mobile** — `notebooks/mosquito_mobile_export.ipynb` does ONNX → TFLite conversion, INT8 quantization, and a latency/size benchmark. This is the path that maps to VectorCam's real phone-based target; the notebook ends with a table comparing the float32 and INT8 models.

**Probe the failure mode:**
```bash
python scripts/test_hard_cases.py \
  --root /path/to/mosquito_cdc \
  --api-url https://mosquito-classifier-207189004007.us-central1.run.app
```

**Check calibration** (is a confidence score trustworthy?):

```bash
python scripts/calibration_check.py \
  --root /path/to/mosquito_cdc \
  --api-url https://mosquito-classifier-207189004007.us-central1.run.app
```

## Author

**Vishwanath Ninganolla** — [GitHub](https://github.com/523vishwanath) · [LinkedIn](https://linkedin.com/in/vishwanathninganolla)