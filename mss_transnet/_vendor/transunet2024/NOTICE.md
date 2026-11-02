# TransUNet 2024 decoder attribution

Adapted from `MultiScaleMaskedTransformerDecoder3d` in the official
Beckschen/3D-TransUNet repository, commit
`9f18182f5f7b26fc81e2f2d70fb5c40ee8a58908`.

Credit: Chen et al., *TransUNet: Rethinking the U-Net architecture design for medical
image segmentation through the lens of transformers*, Medical Image Analysis 97
(2024), 103280. DOI: 10.1016/j.media.2024.103280.

The upstream decoder states Johns Hopkins University copyright and acknowledges
Mask2Former. Original notices are retained in `decoder2d.py` and
`position_encoding.py`. `layerscale.py` contains the unchanged upstream LayerScale
class. The adaptation converts spatial operations to 2D and removes unused runtime
framework wrappers; it preserves the source attention/FFN/prediction-head flow.

`LICENSE` is the upstream TransUNet Apache-2.0 text. `LICENSE.Mask2Former` preserves
the inherited Meta MIT notice, and `LICENSE.DETR` preserves the positional-encoding
ancestor's license. These notices do not assign TransUNet ownership to MSS authors.
