# Data supplied separately

The archive contains source code only. Obtain the datasets from their distributors
under their terms; there is no download or credential embedded in this repository.

## BUSI

Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief 28 (2020),
104863. The complete release has 780 images: 437 benign, 210 malignant, 133 normal.
Use the original `normal/`, `benign/`, and `malignant/` folders, keeping each image
and its `_mask*.png` files together. Multiple masks are merged by pixelwise union.
The preparation command writes a class-stratified image split; it does not establish
patient independence. Existing image IDs/metadata cannot rule out patient overlap.

## BrEaST

Pawlowska et al., *A Curated benchmark dataset for ultrasound based breast lesion
analysis*, Scientific Data 11 (2024), 148. DOI: https://doi.org/10.7937/9WKK-Q141.
Joint reconstruction loads only the 256 primary `caseNNN.png` images. Tumor masks,
other-structure masks, diagnosis and clinical metadata are not loaded as targets.

## BUS-BRA (optional)

Gomez-Flores et al., *BUS-BRA: A breast ultrasound dataset for assessing computer-aided
diagnosis systems*, Medical Physics 51 (2024), 3110–3123. DOI: https://doi.org/10.1002/mp.16812.
The complete release has 1,875 lesion scans, 1,268 benign and 607 malignant.
Use its `bus_data.csv`, `Images/`, and `Masks/` directories. The included preparation
function groups patients by the `Case` field. A newly generated hold-out is not the
five-fold manifest used in the archived experiments; regenerate/preserve the intended
fold manifest separately if reproducing a particular experiment.

## Manifest format

The loader needs `sample_id,class_name,image_path,mask_paths,split`. Preparation also
records dataset/group IDs, image dimensions and hashes. Use absolute paths on your
machine; separate multiple mask paths with `;`. Valid split values are `train` and
`test`. Do not upload manifests containing sensitive paths or clinical metadata.

The optional SSL image CSV contains only `image_path`. Its default collector selects
training images plus primary BrEaST images. Including held-out test images makes the
experiment transductive even when their annotations are never read; document that
choice explicitly. Keep test patients outside supervised training and never select
checkpoints or thresholds using their labels during training.
