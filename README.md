# Proposed breast ultrasound framework — deeper heads, 30% masking

A source-only implementation of TransUNet with segmentation, masked reconstruction,
tumor classification and relative tumor-area estimation. The default version has
independent deeper bottleneck heads: **768 → 256 → 128 → output**.

**No trained weights, pretrained weights, datasets or experiment outputs are included.**
The reconstruction masking default is **30%** in SSL and joint training: 59 of 196
randomly distributed 16×16 patches at 224×224 resolution (30.102% after rounding).

<p align="center">
  <img src="docs/figures/proposed_framework.svg" alt="Proposed framework figure placeholder" width="950">
</p>

*Figure placeholder: replace `docs/figures/proposed_framework.svg` with the final illustration.*

## Installation

Python 3.10+ is required; the package is checked using the existing Python 3.11 /
PyTorch 2.10 environment. First install a compatible PyTorch and torchvision build
for your CPU/CUDA environment. From this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -B scripts/smoke_test.py
```

Use the editable installation from this source checkout: TransUNet's bundled source
is resolved from `external/TransUNet`. The smoke check uses synthetic tensors and
random initialization; it needs neither a GPU, data nor downloaded weights.

## Prepare data

Obtain BUSI and BrEaST separately. See [data layout and references](docs/DATA.md).
Prepare the historical BUSI 95%/5% image hold-out (741/39 with the complete release):

```bash
python -m breast_benchmark.prepare --dataset BUSI \
  --dataset-root /path/to/Dataset_BUSI_with_GT \
  --output data/splits/busi_seed42.csv --test-size 0.05 --seed 42
```

For a patient-level BUS-BRA hold-out:

```bash
python -m breast_benchmark.prepare --dataset BUS-BRA \
  --dataset-root /path/to/BUSBRA \
  --output data/splits/busbra_seed42.csv --test-size 0.05 --seed 42
```

These commands generate new manifests with your local file paths. The BUS-BRA
hold-out command does not recreate the separately executed five-fold experiment.
BUSI patient independence cannot be verified from the image-level identifiers.

## Joint training

Train the default deeper classifier and deeper area head with segmentation and
BrEaST reconstruction, using the [30% preset](configs/joint_mask30.json):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --manifest data/splits/busi_seed42.csv \
  --unlabeled-root /path/to/BrEaST-Lesions_USG-images_and_masks \
  --output-root outputs/joint_deepheads_mask30 --no-resume
```

Defaults: 200 epochs, batch 24, seed 42, AdamW 1e-4, weight decay 1e-4, cosine learning
rate, gradient clipping 0.5, patience 40, AMP on CUDA, 30% masking. Joint loss is
`0.90*segmentation + (classification + area + reconstruction)/30`. No extra image-noise
augmentation is enabled. Override CLI options normally, for example `--batch-size 8`
on a smaller GPU; this changes the physical training batch.

The archive defaults to random initialization so it can run without missing files.
For ImageNet initialization, obtain `R50+ViT-B_16.npz` from the source referenced in
`external/TransUNet/README.md`, place it under `weights/`, and add `--pretrained`.
That initialization file is intentionally excluded from this ZIP and `.gitignore`
keeps weight files out of the repository. Random initialization is not a reproduction
of the historical ImageNet-initialized result.

The default run writes `config.json`, `history.csv`, best/last checkpoints and per-image
segmentation, classification and area CSVs under the selected output root. These files
are created only when you train; they are not part of this source archive.

## Optional SSL followed by joint training

For a two-stage experiment, collect training image paths without labels or masks:

```bash
python scripts/make_ssl_manifest.py \
  --labeled-manifest data/splits/busi_seed42.csv \
  --unlabeled-root /path/to/BrEaST-Lesions_USG-images_and_masks \
  --output data/splits/ssl_images.csv

CUDA_VISIBLE_DEVICES=0 python -m breast_benchmark.pretrain \
  --image-manifest data/splits/ssl_images.csv \
  --output-root outputs/ssl_deepheads_mask30 --mae-mask-ratio 0.30

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --manifest data/splits/busi_seed42.csv \
  --unlabeled-root /path/to/BrEaST-Lesions_USG-images_and_masks \
  --initial-checkpoint outputs/ssl_deepheads_mask30/best_model.pt \
  --output-root outputs/ssl_then_joint_mask30 --no-resume
```

SSL trains the full shared encoder plus reconstruction modules and freezes supervised
heads and segmentation decoder. Joint training then unfreezes everything. The SSL
runner is a portable utility supplied with this export; it does not reproduce the
older restricted-encoder experiments. It uses one GPU per run and does not silently
start training on all GPUs. The JSON SSL file records the default settings; the SSL
runner uses CLI arguments, while `scripts/train.py` loads the joint JSON preset.

Repeat `--labeled-manifest` to include training images from BUS-BRA or another manifest.
By default only `train` rows enter SSL. Use `--split all` only when intentionally
performing transductive evaluation; document any unlabeled test-image access.

## Evaluate and predict

Evaluate a checkpoint you have trained (use only checkpoints you trust):

```bash
python scripts/evaluate.py \
  --checkpoint /path/to/checkpoints/best_model.pt \
  --manifest data/splits/busi_seed42.csv --output-root outputs/evaluation

python scripts/predict.py \
  --checkpoint /path/to/checkpoints/best_model.pt \
  --image /path/to/ultrasound.png --output-root outputs/prediction
```

For a binary BUS-BRA checkpoint, add `--classes benign malignant`. Set `--model`
consistently when using a single-head ablation. Evaluation reuses the original
per-image segmentation metrics, classification summaries and relative-area errors.
Prediction saves a binary mask at the original image dimensions plus class
probabilities and estimated relative area. It does not run reconstruction at inference.

## Included model variants

| Registry key | Classifier | Area head |
|---|---|---|
| `TransUNetMAESizeDeepHeads` (default) | Deeper MLP | Deeper MLP |
| `TransUNetMAESizeDeepClassifier` | Deeper MLP | Original linear |
| `TransUNetMAESizeDeepArea` | Original linear | Deeper MLP |
| `TransUNetMAESize` | Original linear | Original linear |

Choose a variant with `--model`. All variants retain the shared encoder, original
segmentation decoder and four-layer reconstruction decoder. Detailed head shapes,
parameter counts, targets, masking and objective definitions are in
[ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Results and provenance

This release changes the masking setting to 30% as requested. **No performance is
claimed for this deeper-head, 30% configuration.** Archived deeper-head experiments
used 75% masking; the best recorded BUSI DSC of 86.42% used the original linear heads.
The historical deeper-head variant did not produce that best result.

Model modules, supervised trainer and metrics come from the isolated
`protocol_deeper_heads` implementation. Export edits prune unrelated comparator
registrations, make file messages portable and set documented CLI defaults. Optional
SSL, evaluation, prediction and export checks are portable entry points added for
this release. Source hashes and edits are recorded in `SOURCE_PROVENANCE.json`.

TransUNet's Apache-2.0 license is preserved in `external/TransUNet/LICENSE`.
See [third-party notices](THIRD_PARTY_NOTICES.md). No new license is assigned to
project-specific code by this packaging step.

Upload the contents of this folder to your GitHub repository. Keep `.gitignore`,
replace the framework figure when ready, and supply datasets/weights separately.
