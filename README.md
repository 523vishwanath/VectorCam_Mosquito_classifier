# Mosquito Species + Sex Classifier — VectorCam Portfolio Project

## Try it live

**Web app:** https://523vishwanath.github.io/VectorCam_Mosquito_classifier/
**API directly:** https://mosquito-classifier-207189004007.us-central1.run.app/docs

No mosquito photo handy? Sample specimen images are in [`samples/`](samples/) —
download any of them and upload to the web app above, or hit the API directly:

```bash
curl -X POST -F "file=@samples/aedes_aegypti_female.jpg" \
  https://mosquito-classifier-207189004007.us-central1.run.app/predict
```

| Image | Species | Sex |
|---|---|---|
| `aedes_aegypti_female.jpg` | *Aedes aegypti* | female |
| `anopheles_arabiensis_female.jpg` | *Anopheles arabiensis* | female |
| `anopheles_coluzzi_female.jpg` | *Anopheles coluzzii* | female |
| `anopheles_albimanus_male.jpg` | *Anopheles albimanus* | male |
| `anopheles_freeborni_male.jpg` | *Anopheles freeborni* | male |



The service scales to zero when idle — the first request after a period of
inactivity takes a few extra seconds to wake up. This is expected, not a bug.

A two-head EfficientNet-B0 classifier that identifies mosquito **species** (7 classes)
and **sex** (male/female) from a single image, built as a portfolio project supporting
an application to the Johns Hopkins Center for Bioengineering Innovation and Design's
VectorCam platform (AI-enabled mobile mosquito identification for malaria control).

**Live demo:** https://mosquito-classifier-207189004007.us-central1.run.app/docs
**Training dashboard (Comet ML):** https://www.comet.com/vishwanath-reddy/vectorcam-mosquito

## Why this project

VectorCam's underlying model, VectorBrain (Li et al., 2024), performs concurrent
multi-task classification of mosquito species, sex, and abdominal status from a
single specimen image, using a shared CNN backbone with independent output heads.
This project replicates that architecture on public data to validate the approach
end-to-end: data pipeline, multi-task training, evaluation, and cloud deployment.

Reference: Li, D., et al. (2024). *Towards Transforming Malaria Vector Surveillance
Using VectorBrain.* https://doi.org/10.21203/rs.3.rs-4462833/v1

## Dataset

CDC/MR4 mosquito image collection (Dryad, doi:10.5061/dryad.z08kprr92), 740 images
across 7 species from 3 genera (Anopheles, Aedes, Culex), both sexes. Labels are
embedded in filenames (`genus_species_sex_strain_imagenumber.jpg`) and parsed into
a structured `labels.csv`.

Note: abdominal status (the third head in VectorBrain) has no public dataset
equivalent — this is proprietary field data VectorCam collected themselves. This
project validates the species + sex heads; the architecture is designed so a third
head (abdomen status, masked to female-only samples) can be added directly once
field data is available, using the same conditional-loss pattern already implemented
for handling missing labels.

## Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained), shared trunk
- **Heads:** two independent linear classifiers (species: 7-way, sex: 2-way)
- **Training:** three-stage progressive unfreezing
  1. Frozen backbone, heads only
  2. Unfreeze last 3 blocks, fine-tune at lower LR
  3. Full unfreeze, fine-tune at very low LR with early stopping
- **Loss:** class-weighted (sqrt inverse frequency) cross-entropy with label smoothing,
  summed across both heads
- **Inference:** test-time augmentation (flip variants averaged)

## Results

| Metric | Score |
|---|---|
| Species macro-accuracy (test, TTA) | 89.4% |
| Sex macro-accuracy (test, TTA) | 92.6% |

Confusion matrix: `confusion_matrices.png` (in repo)

**Known limitation:** confusion concentrates between *An. arabiensis* and
*An. coluzzii* — sibling species within the *An. gambiae* complex that are
difficult to distinguish morphologically even for trained entomologists. This
mirrors the difficulty pattern reported in the VectorBrain paper itself, and is
treated as an honest limitation rather than something to be optimized away.

## Repo structure

```
├── README.md
├── .gitignore
├── notebooks/
│   ├── mosquito_multihead_training.ipynb   # full training notebook (Colab, A100)
│   └── mosquito_mobile_export.ipynb        # ONNX/TFLite export + latency benchmark
├── scripts/
│   ├── test_hard_cases.py                  # tests the live API on known-hard species pairs
│   └── calibration_check.py                # checks confidence calibration on the live API
├── mosquito_deploy/
│   ├── app.py                              # FastAPI inference service
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── README.md                           # deployment-specific instructions
│   └── mosquito_multihead_effnetb0_final.pt   # trained weights — see note below
├── results/
│   ├── confusion_matrices.png
│   ├── class_maps.json
│   └── labels.csv
```

**Note on the checkpoint file**: `mosquito_multihead_effnetb0_final.pt` (~16MB) is
excluded from git via `.gitignore` by default since binary model weights don't
belong in a normal git history. Either commit it directly if your repo host allows
files this size (GitHub does, up to 100MB, just less ideal for version control),
or use Git LFS (`git lfs track "*.pt"`), or link to it externally (e.g. a release
asset) and note that in this README.

## Running it yourself

**Train:** open `notebooks/mosquito_multihead_training.ipynb` in Colab (A100 runtime
recommended), point `ROOT` at your unzipped dataset + `labels.csv`, run top to bottom.

**Deploy:** see `mosquito_deploy/README.md` for the Cloud Run deployment steps.

**Export for mobile:** open `notebooks/mosquito_mobile_export.ipynb` in Colab (CPU
runtime is enough), upload your trained checkpoint when prompted.

**Test the live API against known-hard cases:**
```bash
python scripts/test_hard_cases.py --root /path/to/mosquito_cdc --api-url https://mosquito-classifier-207189004007.us-central1.run.app
```

**Check confidence calibration:**
```bash
python scripts/calibration_check.py --root /path/to/mosquito_cdc --api-url https://mosquito-classifier-207189004007.us-central1.run.app
```

**Query the live API directly:**
```bash
curl -X POST -F "file=@your_mosquito_image.jpg" \
  https://mosquito-classifier-207189004007.us-central1.run.app/predict
```

## Production considerations (not implemented here, noted for completeness)

- **Mobile deployment:** ONNX export → TFLite/CoreML conversion → INT8 quantization,
  matching VectorCam's actual phone-based deployment target. Latency/size benchmarks
  pending.
- **Calibration:** confidence scores should be validated against actual accuracy
  before being used as a "defer to human" signal in a real field tool.
- **Abdomen status head:** requires field data; architecture supports adding it
  directly via the same masked multi-task loss pattern used for the existing heads.

## Author

Vishwanath Ninganolla — github.com/523vishwanath — linkedin.com/in/vishwanathninganolla
