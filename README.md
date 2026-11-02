# MSS-Trans-Net

**Multitask semi-supervised learning with transformer-based network for breast
cancer diagnosis in ultrasound scans**

Mehreen Kanwal and Yunsik Son, Department of Computer Science and Artificial
Intelligence, Dongguk University, Seoul, Republic of Korea.

## Description

MSS-Trans-Net is a segmentation-prioritized multitask framework for breast
ultrasound images. A shared hybrid CNN–transformer encoder (R50–ViT-B/16) feeds
four branches:

- lesion segmentation: CNN decoding followed by prediction-guided transformer
  refinement;
- diagnostic classification: normal, benign or malignant;
- auxiliary relative-area regression;
- masked-image reconstruction.

Training has two phases:

1. **Self-supervised learning (SSL).** The ImageNet-initialized encoder learns
   to reconstruct randomly masked patches of training images. No annotations
   are used.
2. **Semi-supervised multitask learning (SML).** All four branches are then
   trained jointly, with segmentation as the primary objective.

![MSS-Trans-Net framework](docs/figures/framework.png)

*Architecture of MSS-Trans-Net and its two training phases. A shared
CNN–transformer encoder supplies a CNN reconstruction decoder, a segmentation
branch combining CNN decoding with transformer refinement, and independent
classification and relative-area regression heads. Green arrows mark the
self-supervised phase (masked training images, no annotations); blue arrows mark
the semi-supervised multitask phase. ρ is the predicted relative lesion area and
𝒵 the diagnostic class. The example ultrasound image and mask are from the BUSI
dataset (Al-Dhabyani et al., 2020).*

This repository provides a PyTorch implementation with:

- data preparation for BUSI (five-fold evaluation) and BUS-BRA (patient-disjoint
  test set);
- two-phase training with validation-based checkpoint selection;
- evaluation, five-fold pooling, and cross-dataset transfer to BUS-BRA (zero-shot,
  limited-target fine-tuning and full-target training);
- prediction for a single image.

Datasets are not included; see [Dataset information](#dataset-information).

## Dataset information

| | BUSI | BUS-BRA |
|---|---|---|
| Content | 780 breast ultrasound images (133 normal, 437 benign, 210 malignant) from 600 female patients, with reference lesion masks. Acquired in 2018 at Baheya Hospital, Cairo, Egypt | 1,875 lesion images (1,268 benign, 607 malignant) from 1,064 patients, with reference lesion masks, BI-RADS categories and patient (`Case`) identifiers |
| Source | https://scholar.cu.edu.eg/?q=afahmy/pages/dataset (archive `Dataset_BUSI.zip`) | https://doi.org/10.5281/zenodo.8231412 (version 1.0, archive `BUSBRA.zip`) |
| Terms | Publicly released by the dataset authors. Cite Al-Dhabyani et al. (2020) | Creative Commons Attribution 4.0 International. Cite Gómez-Flores et al. (2024) |
| Use here | Five-fold training and evaluation | Cross-dataset evaluation |

The datasets are not redistributed in this repository. Download them from the
sources above, use them under their terms, and keep the original folder layout:

```text
Dataset_BUSI_with_GT/
  normal/      normal (1).png,     normal (1)_mask.png, ...
  benign/      benign (1).png,     benign (1)_mask.png, ...
  malignant/   malignant (1).png,  malignant (1)_mask.png, ...
BUSBRA/
  bus_data.csv
  Images/      bus_0001-l.png, ...
  Masks/       mask_0001-l.png, ...
```

Some BUSI images have more than one mask (`_mask_1.png`, ...); these are merged
by pixelwise union. BUSI has no patient identifiers, so its folds are split by
image. BUS-BRA splits keep all images of a patient in the same partition.

## Code information

| Path | Purpose |
|---|---|
| `mss_transnet/model.py` | Network: encoder, segmentation branch, classification and area heads, reconstruction branch |
| `mss_transnet/losses.py` | Segmentation structure loss, hidden-patch reconstruction loss, class weights, area decoding |
| `mss_transnet/data.py` | Image and mask loading, augmentation, and split preparation (`python -m mss_transnet.data`) |
| `mss_transnet/engine.py` | Two-phase training with validation-based checkpoint selection (`python -m mss_transnet.engine`) |
| `mss_transnet/evaluate.py` | Test evaluation, five-fold pooling and five-model averaging (`python -m mss_transnet.evaluate`) |
| `mss_transnet/metrics.py` | Segmentation and classification metrics |
| `configs/default.json` | Training configuration (also packaged as `mss_transnet/configs/default.json`) |
| `mss_transnet/_vendor/` | Third-party network components (see `NOTICE.md`) |
| `scripts/run_busi.py` | Five-fold BUSI training, evaluation and pooling |
| `scripts/run_busbra.py` | BUS-BRA zero-shot, limited-target and full-target experiments |
| `scripts/predict.py` | Prediction for one image |
| `tests/` | Unit tests (`test_method.py`) and a CPU integration check (`run_end_to_end.py`) on synthetic data |

## Requirements

- Python 3.10 or later.
- PyTorch 2.x. A CUDA GPU is needed for training; `requirements-pinned.txt` pins
  PyTorch 2.10.0, so install its CUDA 12.8 build.
- numpy, Pillow, scipy, scikit-learn and ml-collections (`requirements.txt`).
- pytest, for the tests.
- The ImageNet-pretrained `R50+ViT-B_16.npz` hybrid weights distributed with
  TransUNet (https://github.com/Beckschen/TransUNet), saved as
  `weights/R50+ViT-B_16.npz`.

## Usage instructions

### 1. Install

```bash
git clone https://github.com/kanwaal/MSS-TransNet.git
cd MSS-TransNet
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# Install the PyTorch build for your CUDA version first (https://pytorch.org).
python -m pip install -r requirements.txt     # or requirements-pinned.txt
python -m pip install -e ".[test]"
python -m pytest -q
```

### 2. Add the ImageNet initialization

Place `R50+ViT-B_16.npz` in `weights/`. Training commands load it with
`--pretrained weights/R50+ViT-B_16.npz`.

### 3. Prepare the data splits

```bash
python -m mss_transnet.data --dataset BUSI --root /path/to/Dataset_BUSI_with_GT \
  --output data/busi_folds --seed 42
python -m mss_transnet.data --dataset BUS-BRA --root /path/to/BUSBRA \
  --output data/busbra --seed 42
```

- **BUSI** produces `fold_01.csv` to `fold_05.csv`. Each fold holds out 156
  test images (20%). Its remaining 624 images are split into 593 training and
  31 validation images. Every image is tested exactly once across the five
  folds.
- **BUS-BRA** produces `full.csv` and `limited.csv`. Both share one
  patient-disjoint 20% test set. `full.csv` trains on all remaining patients;
  `limited.csv` trains on 20% of them. Each setting holds out 5% of its training
  patients for validation.

Each manifest comes with a `.json` file recording image and patient counts and
the manifest's SHA-256 hash.

### 4. BUSI: five-fold training and evaluation

```bash
python scripts/run_busi.py --manifests data/busi_folds --output runs/busi \
  --pretrained weights/R50+ViT-B_16.npz
```

For each fold, the script runs SSL, then SML, then evaluates the selected
checkpoint on that fold's test images. Results go to two places:

- `runs/busi/fold_XX/test/`: per-image CSV files and `summary.json` for each
  fold;
- `runs/busi/pooled/pooled_summary.json`: segmentation on the 647 lesion images
  and classification on all 780 images.

To run a single fold:

```bash
python -m mss_transnet.engine --manifest data/busi_folds/fold_01.csv \
  --output runs/fold_01 --pretrained weights/R50+ViT-B_16.npz
python -m mss_transnet.evaluate single --checkpoint runs/fold_01/sml/best.pt \
  --manifest data/busi_folds/fold_01.csv --output runs/fold_01/test
```

### 5. BUS-BRA: cross-dataset evaluation

```bash
# Zero-shot: evaluate the five BUSI fold models separately
python scripts/run_busbra.py --setting zero-shot --manifest data/busbra/full.csv \
  --output runs/busbra_zero --source-checkpoints \
  runs/busi/fold_01/sml/best.pt runs/busi/fold_02/sml/best.pt \
  runs/busi/fold_03/sml/best.pt runs/busi/fold_04/sml/best.pt \
  runs/busi/fold_05/sml/best.pt

# Limited-target fine-tuning of each BUSI fold model
python scripts/run_busbra.py --setting limited --manifest data/busbra/limited.csv \
  --output runs/busbra_limited --source-checkpoints \
  runs/busi/fold_01/sml/best.pt runs/busi/fold_02/sml/best.pt \
  runs/busi/fold_03/sml/best.pt runs/busi/fold_04/sml/best.pt \
  runs/busi/fold_05/sml/best.pt

# Full-target training from ImageNet initialization
python scripts/run_busbra.py --setting full-target --manifest data/busbra/full.csv \
  --output runs/busbra_full --pretrained weights/R50+ViT-B_16.npz
```

Zero-shot and limited-target results are averaged over the five models in
`target_summary.json`.

### 6. Predict one image

```bash
python scripts/predict.py --checkpoint runs/busi/fold_01/sml/best.pt \
  --image /path/to/image.png --output outputs/example
```

The output folder contains three files:

- `mask_224.png`: the lesion mask at 224 × 224;
- `mask_original_size.png`: the same mask at the input image's size;
- `prediction.json`: class probabilities and the relative lesion area, in
  percent.

Add `--binary` to report benign/malignant probabilities from a three-class BUSI
model.

### Output conventions

Every run directory contains:

- `experiment.json`;
- for each phase, a `best.pt`, `history.json` and `selection.json`.

Metric values in JSON files are fractions between 0 and 1, except confusion
matrices labelled `row_percent`, which are percentages.

## Methodology

**Preprocessing and augmentation.**

- RGB images are scaled to [0, 1] and resized to 224 × 224 with bilinear
  interpolation.
- Masks are resized with nearest-neighbour interpolation.
- During training, each image–mask pair is flipped horizontally and
  vertically (each with probability 0.5), then rotated by 0°, 90°, 180° or
  270°. Image and mask receive the same transformation.
- ImageNet channel normalization is applied inside the model.

**Network.**

- **Encoder.** R50–ViT-B/16 hybrid encoder (12 transformer layers, 12 heads,
  width 768), giving 196 tokens of dimension 768. It is initialized from
  ImageNet weights.
- **Segmentation branch.** A CNN decoder with encoder skip connections
  produces features at 28, 56, 112 and 224 pixels. A masked transformer decoder
  then refines them: three layers, width 192, 8 heads, feed-forward width 1536,
  LayerScale, and one learnable foreground query. Each layer restricts
  cross-attention to the foreground predicted by the previous layer. A 0.5
  threshold on the sigmoid output gives the binary mask.
- **Classification and relative-area heads.** Mean-pooled tokens →
  Dropout(0.2) → 256 → GELU → Dropout(0.2) → 128 → GELU → Dropout(0.2) → 64 →
  output. The output is 3 classes for BUSI, 2 for BUS-BRA, and 1 value for the
  area head. The two heads have separate weights.
- **Reconstruction branch.** 59 of the 196 16 × 16 patches (30.1%) are chosen at
  random and set to black. The whole masked image passes through the encoder. A
  CNN decoder (four 2× upsampling stages with 256, 128, 64 and 32 channels, then
  a sigmoid RGB output) reconstructs the image.

**Objectives.**

- *Segmentation:* boundary-weighted binary cross-entropy plus weighted IoU,
  with pixel weights w = 1 + 5·|AvgPool₃₁(y) − y|.
- *Reconstruction:* mean squared error over the hidden-patch pixels only.
- *Classification:* cross-entropy with inverse-frequency class weights computed
  on the training split.
- *Relative area:* Smooth L1 loss on log(1 + a), where a is the lesion's
  percentage of the original-resolution image. Predictions are mapped back with
  exp(clip(t, 0, log 101)) − 1.
- *SML objective:* 0.90 × segmentation + (classification + area +
  reconstruction) / 30.

**Training.**

- **SSL** trains the encoder and reconstruction branch for 50 epochs on training
  images, without masks or labels. The checkpoint with the lowest validation
  reconstruction loss is kept. Validation uses fixed random masks.
- **SML** starts from the SSL checkpoint and trains all branches for up to 150
  epochs. It stops early after 15 epochs without validation improvement. The
  checkpoint with the lowest validation objective is kept.
- **Optimizer:**
  - AdamW, learning rate 1e-4, weight decay 1e-4;
  - cosine schedule, restarted in each phase;
  - batch size 24;
  - mixed precision on CUDA;
  - gradient-norm clipping at 0.5;
  - seed 42.
- Validation images never contribute gradients. Test images are read only
  after the final checkpoint has been selected.

**Evaluation.**

- **BUSI.** Predictions from the five held-out folds are pooled, so each image
  is evaluated once.
  - Segmentation uses the 647 lesion images. Metrics are Dice, IoU, precision,
    sensitivity, specificity and pixel accuracy, given as the mean and standard
    deviation over images.
  - Three-class classification uses all 780 images. Metrics are accuracy and
    macro-averaged precision, recall, specificity and F1.
- **BUS-BRA** (benign vs. malignant) uses the common patient-disjoint test set:
  - **Zero-shot:** each of the five BUSI fold models is evaluated separately.
    The malignant probability is renormalized over the two lesion classes,
    p_M / (p_B + p_M).
  - **Limited-target fine-tuning:** each BUSI fold model is fine-tuned on
    `limited.csv` with the encoder frozen. All four branches are trained, with
    no additional SSL.
  - **Full-target training:** one model is trained on `full.csv` from ImageNet
    initialization, with SSL followed by SML and without BUSI weights.
  - Zero-shot and limited-target results are averaged over the five models.
    This covers segmentation means, row-normalized confusion matrices and ROC
    curves; predictions are not ensembled.

All settings are in `configs/default.json`.

## Citations

If you use this code, please cite:

> Kanwal M, Son Y. MSS-Trans-Net: Multitask semi-supervised learning with
> transformer-based network for breast cancer diagnosis in ultrasound scans.
> Manuscript under review, PeerJ Computer Science, 2026.

Datasets:

- Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. 2020. Dataset of breast ultrasound
  images. *Data in Brief* 28:104863. https://doi.org/10.1016/j.dib.2019.104863
- Gómez-Flores W, Gregorio-Calas MJ, de Albuquerque Pereira WC. 2024. BUS-BRA: a
  breast ultrasound dataset for assessing computer-aided diagnosis systems.
  *Medical Physics* 51(4):3110–3123. https://doi.org/10.1002/mp.16812
  (dataset: https://doi.org/10.5281/zenodo.8231412)

The encoder and decoder components build on TransUNet (Chen et al., 2024,
*Medical Image Analysis* 97:103280, https://doi.org/10.1016/j.media.2024.103280).

BibTeX entries are in `REFERENCES.bib`; citation metadata is in `CITATION.cff`.

## License and contribution guidelines

The code in this repository is released under the Apache License 2.0 (`LICENSE`).
Third-party components in `mss_transnet/_vendor/` keep their original licenses,
as listed in `NOTICE.md`. The datasets and the ImageNet initialization weights
are not covered by this license and remain under their providers' terms.

Contributions are welcome:

- Report bugs or ask questions through GitHub issues. Include your environment
  and the command you ran.
- Submit changes as pull requests with a short description, and run
  `python -m pytest -q` first.
- By contributing, you agree that your contribution is licensed under the
  Apache License 2.0.
