# Reproducibility notes

## Environment

* Apple M3, 8 GB unified memory, macOS (Darwin 25.6).
* Python 3.12.13 in a uv virtual environment (`make setup`).
* Direct dependencies pinned in `requirements.txt`; the full environment
  (139 packages) pinned in `requirements-lock.txt`
  (`uv pip install -r requirements-lock.txt` recreates it).
* PyTorch 2.15.0. Training and inference run on the Apple GPU through MPS
  when it is available, otherwise CUDA, otherwise CPU
  (`angiolens.models.unet.pick_device`). Every training log records the
  device and library versions (`results/train_logs/<run>.json`).

## Data provenance

* `make data` downloads one file from Zenodo and refuses to extract it unless
  its size, the MD5 published by Zenodo and the pinned SHA-256 all match (see
  `docs/data_card.md`). Extraction is allow-listed (PNG and JSON under
  `arcade/syntax/` only) and guarded against path traversal, symlinks and zip
  bombs. Nothing from the archive is executed or unpickled; arrays are loaded
  with `allow_pickle=False` and checkpoints with `torch.load(weights_only=True)`.
* `results/splits.csv` lists every image id, its split, its duplicate group and
  whether it was used, so the exact train, validation and test sets can be
  rebuilt or audited without the code.
* Synthetic images are regenerated bit for bit from their seeds
  (`results/synth_summary.json` records the seeds and generator settings).

## Randomness

* Each config sets one seed. It seeds Python, NumPy and PyTorch, the data
  loader shuffle, the per-sample augmentation (a NumPy generator seeded with
  seed, epoch and sample index) and the low-data subset selection.
* MPS kernels are not guaranteed to be bitwise deterministic, so retraining on
  the Apple GPU gives close but not identical weights. Everything after
  training (thresholds, metrics, bootstrap intervals with fixed seeds,
  figures) is deterministic given the checkpoints.
* Runs have a wall-clock guard (`max_minutes` in each config). The runs that
  hit it are marked `stopped_by_time_limit` in their training log, and the
  checkpoint used is always the best validation epoch, not the last one.

## What decides what

| Decision | Made on | Never uses |
|---|---|---|
| Frangi rule, cutoff and minimum component size | validation split | test split |
| U-Net checkpoint (best epoch) | validation Dice at threshold 0.5 | test split |
| U-Net probability threshold | validation Dice, grid 0.10 to 0.90 | test split |
| Synthetic-only model checkpoint and threshold | synthetic validation set | any ARCADE label |
| Primary model for failure analysis | validation Dice | test split |
| Near-duplicate threshold | visual review of the top ranked pairs (no labels) | metrics |

## Outputs

| Path | Content |
|---|---|
| results/data_qa.md, data_qa.json, data_qa_per_image.csv | dataset checks |
| results/splits.csv, split_summary.json | split with exclusions |
| results/synth_summary.json | synthetic set seeds and statistics |
| results/frangi_tuning.json | the full validation grid and the frozen setting |
| results/train_logs/ | per-epoch logs, configs, environment and timing of every run |
| results/thresholds/ | validation threshold search per U-Net run |
| results/metrics/ | per-image metrics, summaries with CIs, paired tests, failure analysis |
| results/figures/, results/failures/ | plots and overlays |
| results/benchmark.json | inference latency |
| results/review_queue.csv | uncertainty-ranked review queue |

Checkpoints (`checkpoints/`) and all data (`data/`) are gitignored.

## Re-running

```bash
make setup            # once
make all              # data, QA, split, synthetic set, baseline, 5 runs, predictions, evaluation, benchmark, tests
```

Single steps can be rerun on their own (`make baseline`, `make train-scratch`,
`make evaluate`, ...). `make smoke` trains a tiny model on 16 images to check
the training code in about a minute.
