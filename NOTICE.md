# Source and attribution

The project implements the user-supplied updated MSS-Trans-Net manuscript.
The archive's SHA256 and relevant member/line locations are preserved in
`SOURCE_PROVENANCE.json`. The manuscript itself, dataset illustrations and clinical
images are not republished in this implementation archive.

## TransUNet 2024 acknowledgement

The encoder–decoder foundation is credited to Jieneng Chen and the TransUNet authors:
*TransUNet: Rethinking the U-Net architecture design for medical image segmentation
through the lens of transformers*, Medical Image Analysis 97 (2024), 103280.
DOI: https://doi.org/10.1016/j.media.2024.103280.

The hybrid encoder and original CNN decoder use unmodified source from
[Beckschen/TransUNet](https://github.com/Beckschen/TransUNet), pinned to commit
`02ef0010b36eb8328b5e689eadaf613602edf9b8`. The source's original attribution and
Apache-2.0 license are retained in `mss_transnet/_vendor/transunet/`.

The masked transformer decoder is adapted directly from the authors' companion
[Beckschen/3D-TransUNet](https://github.com/Beckschen/3D-TransUNet) repository,
commit `9f18182f5f7b26fc81e2f2d70fb5c40ee8a58908`. The source-derived runtime is in
`mss_transnet/_vendor/transunet2024/`; exact unmodified source snapshots, configuration,
hashes and a change manifest are under `third_party/transunet2024/`.

The 2D adaptation retains the authors' transformer-decoder architecture; dimensional
operations and the binary breast-task interface are identified in
`docs/TRANSUNET_2024.md`. It does not claim an unchanged volumetric nnU-Net backbone.
MSS-specific additions are reconstruction, diagnostic/area heads and the training
procedure, rather than ownership of the TransUNet encoder–decoder itself.

The original Johns Hopkins University notice in the decoder and Facebook/Meta
position/Mask2Former notices are preserved. Inherited license texts are included:
`LICENSE` (TransUNet Apache-2.0), `LICENSE.Mask2Former` (Meta MIT), and `LICENSE.DETR`
under the runtime vendor directory and source snapshot directory. Full citations are
in `REFERENCES.bib` and `CITATION.cff`. This packaging step does not assign a new
license or authorship to another author's source or to project-specific research code.

Dataset references:

* Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief 28 (2020),
  104863. DOI: https://doi.org/10.1016/j.dib.2019.104863.
* Gomez-Flores et al., *BUS-BRA: A breast ultrasound dataset for assessing
  computer-aided diagnosis systems*, Medical Physics 51 (2024), 3110–3123.
  DOI: https://doi.org/10.1002/mp.16812; dataset DOI: https://doi.org/10.5281/zenodo.8231412.

Obtain the original datasets under their distributors' terms. No dataset download,
pretrained parameter file, trained checkpoint, patient metadata or result artifact
from a previous experiment is included in this source package.
