# Architecture and training configuration

The default registry key is `TransUNetMAESizeDeepHeads`. The deeper bottleneck heads
operate on the final transformer representation; this variant does not add residual
blocks to the R50 encoder or transformer layers to the shared encoder.

1. RGB input at 224×224, scaled to [0,1], then ImageNet channel normalization.
2. R50–ViT-B/16 hybrid encoder, 14×14 tokens of dimension 768, three skip levels.
3. Original TransUNet segmentation decoder and one-channel segmentation logits.
4. Mean token pooling to 768 features for two independent supervised MLP heads:

```text
Classification: 768 → Dropout(0.2) → Linear(256) → GELU → Dropout(0.2)
                    → Linear(128) → GELU → Dropout(0.2) → Linear(classes)
Area:           768 → Dropout(0.2) → Linear(256) → GELU → Dropout(0.2)
                    → Linear(128) → GELU → Dropout(0.2) → Linear(1)
```

Both retain AdaptiveAvgPool2d(1) and Flatten before the first dropout. BUSI uses
normal/benign/malignant logits; BUS-BRA uses benign/malignant after resizing only
the output projection. The area target is log1p(percentage of foreground pixels in
the original union mask). It estimates relative 2D image area, not a physical size.
The area output projection starts at zero; hidden layers begin receiving gradients
after the output projection changes.

The reconstruction decoder projects visible 768-dimensional encoder tokens to 256,
restores masked locations with learned tokens and positional embeddings, and applies
four transformer decoder layers (8 attention heads, feedforward width 1024), LayerNorm
and a 768-value RGB patch projection. Masked patch MSE uses clean RGB targets.

At 30% requested masking, `round(196×0.30)=59` patches are masked and 137 remain
visible. Patches are sampled without replacement over the image, not as one contiguous
rectangle. Each 16×16 pixel region is set to zero after normalization before hybrid
embedding; only visible embeddings enter the reconstruction transformer encoder.
The measured pixel fraction is 59/196 ≈ 30.102%. Full unmasked images enter the
supervised segmentation/classification/area path. The random mask is resampled per
image and forward pass, in both optional SSL and joint reconstruction.

| Variant | Total parameters (3 classes) |
|---|---:|
| Original linear heads, TransUNetMAESize | 108,883,221 |
| Deeper classifier only | 109,111,061 |
| Deeper area only | 109,112,341 |
| Both deeper heads (default) | 109,340,181 |

The default classifier contains 230,147 parameters and the area head 229,889.
The shared segmentation network contains 105,275,921 parameters. These counts
include weights created at runtime; no weight tensors are distributed in the ZIP.

Joint loss: `0.90*Lseg + (Lclassification + Larea + Lreconstruction)/30`.
Segmentation uses boundary-weighted BCE plus weighted IoU. Classification uses
inverse-frequency training-class weights. Area uses Smooth L1 on the log1p target.
AdamW lr 1e-4, weight decay 1e-4; 200 epochs; cosine schedule; clip norm 0.5;
physical batch 24; AMP on CUDA; seed 42; patience 40 using total training loss.
There is no separate validation split in this exported historical trainer. The held-out
partition is evaluated after checkpoint selection. Create a training-only validation
split for new model selection rather than repeatedly tuning against the test cohort.

Optional separate SSL trains only the shared encoder and reconstruction modules,
freezing segmentation decoder and supervised heads. It selects minimum reconstruction
training MSE; subsequent joint training starts a fresh optimizer and unfreezes all
parameters. That optional portable runner is included to support a two-stage workflow;
it is distinct from the archived deeper-head experiment, which used joint training
without separate ultrasound SSL.

This source export sets masking to 30% at the user's request. Historical deeper-head
runs used 75%; the best recorded 86.42% BUSI score used the original linear heads.
Neither score is a measured result for this deeper-head, 30%-masking configuration.
