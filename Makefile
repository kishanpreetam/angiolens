# Angiolens pipeline. `make all` runs every step in order; each step can also
# run on its own. Times are from an Apple M3 with 8 GB of memory.

PY := .venv/bin/python -I
UV ?= uv

.PHONY: help setup data qa split synth baseline train train-scratch train-synth train-finetune \
        train-lowdata predict evaluate benchmark test smoke review review-demo review-screenshot all

help:
	@echo "make setup      create .venv (Python 3.12) and install pinned packages"
	@echo "make data       download ARCADE (452 MB), verify checksums, extract, build masks"
	@echo "make split      data QA report and the leakage-safe split"
	@echo "make synth      generate 1,300 synthetic angiograms with exact masks"
	@echo "make baseline   Frangi baseline, threshold tuned on validation"
	@echo "make train      five U-Net runs (about 50 minutes in total)"
	@echo "make predict    thresholds tuned on validation, test predictions"
	@echo "make evaluate   test metrics, confidence intervals, figures, failure cases"
	@echo "make benchmark  inference latency on the Apple GPU and CPU"
	@echo "make test       unit tests"
	@echo "make review     open the review tool on the uncertainty-ranked queue"
	@echo "make all        everything above, in order"

setup:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python .venv/bin/python -r requirements.txt
	$(UV) pip install --python .venv/bin/python -e . --no-deps

data:
	$(PY) -m angiolens.data.download
	$(PY) -m angiolens.data.arcade

qa:
	$(PY) -m angiolens.data.qa

split: qa
	$(PY) -m angiolens.data.split

synth:
	$(PY) -m angiolens.synth.dataset 1200 100

baseline:
	$(PY) -m angiolens.baseline.frangi

train-scratch:
	$(PY) -m angiolens.train configs/unet_scratch.toml

train-synth:
	$(PY) -m angiolens.train configs/unet_synth_pretrain.toml

train-finetune: train-synth
	$(PY) -m angiolens.train configs/unet_synth_finetune.toml

train-lowdata:
	$(PY) -m angiolens.train configs/unet_scratch_n100.toml
	$(PY) -m angiolens.train configs/unet_synth_finetune_n100.toml

train: train-scratch train-finetune train-lowdata

predict:
	$(PY) -m angiolens.predict unet_scratch
	$(PY) -m angiolens.predict unet_synth_only best.pt synth
	$(PY) -m angiolens.predict unet_synth_finetune
	$(PY) -m angiolens.predict unet_scratch_n100
	$(PY) -m angiolens.predict unet_synth_finetune_n100

evaluate:
	$(PY) -m angiolens.eval.evaluate

benchmark:
	$(PY) -m angiolens.benchmark unet_scratch

test:
	$(PY) -m pytest

# Quick end-to-end check of the training code on 16 images (about a minute).
smoke:
	$(PY) -m angiolens.train configs/smoke.toml
	rm -rf checkpoints/smoke results/train_logs/smoke.csv results/train_logs/smoke.json

review:
	$(PY) -m angiolens.review.napari_app --manifest results/review_queue.csv

review-demo:
	$(PY) -m angiolens.review.napari_app --demo 6

# Picture of the review tool for the README, rendered without showing a window.
review-screenshot:
	$(PY) -m angiolens.review.screenshot results/review_queue.csv docs/img/review_tool.png --window

all: data split synth baseline train predict evaluate benchmark test
