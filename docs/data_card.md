# Data card

## Clinical data: ARCADE (syntax subset)

| Field | Value |
|---|---|
| Name | ARCADE: Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs |
| Host | Zenodo record 10390295, version "COCO", published 2023-12-15 |
| DOI | 10.5281/zenodo.10390295 |
| Download URL | https://zenodo.org/api/records/10390295/files/arcade.zip/content (no login) |
| File | arcade.zip, 451,573,813 bytes |
| MD5 (published by Zenodo) | c5b1973ade06f7dff210f878161e1a76 |
| SHA-256 (computed on download, 2026-11-03) | a396cdea7c92c55dc97bbf3dd8e3df517d76872b289a8bcb45513bdb3350837f |
| License | CC0 1.0 Universal (Zenodo license id cc-zero). The data descriptor article is CC BY 4.0. |
| Citation | Popov M, Amanturdieva A, Zhaksylyk N, et al. Dataset for Automatic Region-based Coronary Artery Disease Diagnostics Using X-Ray Angiography Images. Scientific Data 11, 20 (2024). doi:10.1038/s41597-023-02871-z |

CC0 places the data in the public domain, so research use, derived masks and
redistribution are allowed. The citation is given as good practice. The
archive never enters the repository: `make data` downloads it into
`data/raw/` (gitignored), checks size, MD5 and SHA-256, and stops if any of
them differ.

### What is in it

* The archive has two tasks. Only `syntax` is used: 1,500 X-ray coronary
  angiography frames (official split 1,000 train, 200 validation, 300 test),
  512 x 512 pixels, 8-bit gray, extracted from DICOM recordings by the
  dataset authors. 549 training frames are stored as RGB with three identical
  channels; converting them to gray is lossless (checked pixel by pixel).
* Labels: one COCO polygon per coronary segment, using the 25 segment
  definitions of the SYNTAX score (for example 1, 2, 3 for the right coronary
  artery, 5 for the left main, 6 to 8 for the LAD). Frames have 2 to 11
  segments (median 5). The stenosis category exists in the label list but has
  no polygons in this subset.
* The `stenosis` task (different frames, stenosis regions only) is not used and
  is not extracted.
* No patient, study or sequence identifiers are provided.

### How masks are made

The binary vessel mask of a frame is the union of its segment polygons,
filled with Pillow's `ImageDraw.polygon` (fill plus outline). Masks are saved
as 0/255 PNG files in `data/processed/arcade_masks/` and as uint8 arrays in
`data/processed/cache/` for training.

What the mask means matters for every result in this project: it covers the
**annotated SYNTAX segments**, not every visible vessel. Small side branches,
the catheter and contrast outside the named segments are background. A method
that finds every vessel is penalized for it. This is why the Frangi baseline
scores low on precision and why a trained model has to learn the labeling
protocol as well as vessel appearance.

### Quality checks (`make split`, results/data_qa.md)

| Check | Result |
|---|---|
| Images listed in the JSON but missing on disk, or the reverse | 0 |
| Frames without any annotation, empty masks | 0 |
| Unexpected size or mode | 0 (all 512 x 512) |
| Non-binary masks | 0 |
| Exact duplicates (SHA-256 of decoded pixels) | 5 pairs, all inside train |
| Near duplicates (Pearson r of 32 x 32 thumbnails at or above 0.97) | 12 pairs, 1 across splits |

The five exact duplicate pairs are useful in their own right: each pair is the
same frame labeled twice, and the two masks agree with Dice 0.744 to 0.936.
That is a rough indication of how much the reference labels themselves vary.

The near-duplicate threshold was chosen by looking at the highest ranked
pairs (results/figures/near_duplicates.png). It catches frames from the same
run, such as train_720 and val_151, and also a few look-alikes that are only
similar because of identical black borders (train_853 and train_871). Grouping
a look-alike is harmless because grouping only removes images when a group
crosses splits.

| Split | Frames | Mean gray level (mean, sd over frames) | Vessel fraction (median) | Skeleton radius px (median, 95th pct) | Black border fraction (mean) |
|---|---|---|---|---|---|
| train | 1000 | 113.2, 22.0 | 0.0316 | 5.00, 8.94 | 0.0251 |
| val | 200 | 134.6, 13.0 | 0.0331 | 4.47, 9.06 | 0.0137 |
| test | 300 | 124.0, 16.2 | 0.0333 | 4.24, 9.00 | 0.0141 |

The training frames are darker on average than validation and test frames.
The model input is normalized per frame (robust 1st to 99th percentile window
over non-border pixels) to reduce the effect of this shift.

### Split (`results/splits.csv`, `results/split_summary.json`)

ARCADE has no patient identifiers, so a patient-level split is not possible.
The official split is kept, which keeps results comparable with other work,
plus a leakage guard:

1. Images joined by an exact or near duplicate form one group (union-find).
2. A group that crosses splits stays in the earliest split (train, then
   validation, then test); its members elsewhere are excluded.
3. Inside a split, only the first copy of a pixel-identical frame is kept.

| | Train | Validation | Test |
|---|---|---|---|
| Official | 1000 | 200 | 300 |
| Used | 995 | 198 | 300 |

Excluded: train_214, train_612, train_793, train_887, train_972 (second copies
of exact duplicates) and val_116, val_151 (same run as train_720). The test
split is untouched.

Limitation: this catches near-identical frames, not every frame of the same
patient. Different views of one patient can still sit in different splits, so
the test estimate may be slightly optimistic.

## Simulated data: synthetic angiograms (digital phantom)

`angiolens.synth.generator` draws angiogram-like frames with exact masks:
branching vessel trees (smooth random-walk centerlines, tapering radii, side
branches that follow Murray's law), Beer-Lambert attenuation through
cylindrical vessels, a smooth anatomical background, an unlabeled catheter,
detector blur, quantum and electronic noise, and occasional collimator
borders. Root radii (3 to 6.5 px at 512 x 512) bracket the ARCADE skeleton
radius statistics above.

| Set | Frames | Seeds | Use |
|---|---|---|---|
| synthetic train | 1,200 | 1,000,000 + i | pretraining |
| synthetic val | 100 | 2,000,000 + i | model selection for the synthetic-only model |
| synthetic test | 100 | 3,000,000 + i | phantom check in evaluation only |

These are digital phantoms. No physical phantom was imaged, and the
generator is not a validated X-ray simulator; it is a controlled source of
images with exact labels for pretraining and for testing the code.

## Datasets considered and not used

* DCA1 (134 angiograms with expert masks, CIMAT): the host
  (personal.cimat.mx:8181) timed out on 2026-11-03, so it could not be
  downloaded or used as an external test set.
* XCAD: not used in this version.
