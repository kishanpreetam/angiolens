# Model card: Angiolens U-Net

## Intended use

Research and teaching demo of coronary vessel segmentation on X-ray
angiography frames: data curation, training, validation and review tooling.
**Not for clinical use.** It is not a medical device, has not been validated
for diagnosis, treatment planning or any patient care decision, and must not
be used on patient data for those purposes.

## Model

* Architecture: 2D U-Net, 4 pooling stages, 16 base channels (16, 32, 64,
  128, 256), batch norm, ReLU, bilinear upsampling, 1,963,809 trainable
  weights (`src/angiolens/models/unet.py`).
* Input: one gray frame, normalized per frame to [0, 1] with the 1st and 99th
  intensity percentiles of non-border pixels. Trained on 256 x 256 crops,
  applied to full 512 x 512 frames (fully convolutional).
* Output: per-pixel vessel probability; a binary mask with the threshold
  chosen on validation.
* Loss: binary cross-entropy plus soft Dice. AdamW (learning rate 1e-3,
  weight decay 1e-4), linear warm-up then cosine decay, batch size 8.
* Augmentation: random crop, horizontal and vertical flips, gamma, contrast and
  brightness jitter, Gaussian noise, occasional blur.

## Training data

ARCADE syntax subset (CC0): 995 training and 198 validation frames after the
duplicate and leakage checks in `docs/data_card.md`. Optional pretraining on
1,200 synthetic frames from the built-in generator. Variants and their
results are in `docs/validation_report.md`.

## Evaluation data

The official ARCADE test split, 300 frames, never used for any choice.

## Results

ARCADE test split, 300 frames, mean with 95% bootstrap CI
(`results/metrics/summary.json`):

| Model | Dice | clDice | Sensitivity | Precision |
|---|---|---|---|---|
| U-Net, ARCADE (primary, chosen on validation) | 0.685 [0.669, 0.700] | 0.664 [0.648, 0.680] | 0.695 [0.677, 0.712] | 0.701 [0.684, 0.718] |
| U-Net, synthetic pretraining then ARCADE | 0.683 [0.667, 0.698] | 0.665 [0.649, 0.680] | 0.696 [0.678, 0.712] | 0.694 [0.677, 0.711] |
| Frangi baseline, for reference | 0.330 [0.310, 0.349] | 0.303 [0.284, 0.322] | 0.474 [0.445, 0.503] | 0.292 [0.273, 0.312] |

Probability threshold 0.65 for both U-Nets (tuned on validation). Inference
takes 26.8 ms per 512 x 512 frame on an Apple M3 GPU. Typical failures:
catheters marked as vessels, visible vessels outside the annotated segments,
and faint or very wide annotated vessels that are missed
(`docs/validation_report.md`, section 5.4).

## What the model learned, and what it did not

* The reference masks cover the annotated SYNTAX segments only. The model
  learns to segment those segments in the style of these annotators. It
  does not segment every visible vessel, and its output should not be read
  as a complete vessel tree.
* No patient identifiers exist in ARCADE, so the split is by frame with a
  near-duplicate guard. Frames of one patient may appear in both train and
  test, which can make test scores optimistic.
* One dataset, one annotation protocol, no external test set (the DCA1 host
  was unreachable). Performance on other hospitals, scanners, frame rates or
  contrast protocols is unknown.
* Frames are single 2D images; no temporal information from the angiography
  run is used.
* Trained and tested on 512 x 512 frames. Other resolutions need resampling
  so that vessel widths in pixels stay similar.

## Ethical and safety considerations

The data are public (CC0) and consist of image pixels and segment polygons
only; the annotation files carry no patient fields. No patient data were
collected for this project. Any use beyond research needs
clinical validation, a quality management system and regulatory review.
