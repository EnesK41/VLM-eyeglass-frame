# Eyeglass Detection Pipeline

This project finds eyeglasses or sunglasses in images.
It uses Florence-2 for box detection and SAM for mask extraction.

## Input and Output

- Input folder: `data/inputs`
- Output folder: `data/outputs`

Supported input formats: `.jpg`, `.jpeg`, `.png`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

The script reads all images from `data/inputs` and saves visual results to `data/outputs`.

*Intended to make it so that it only segments the lenses of the eyeglass but failed in doing so.

## Limitations and Design Choices

- Transparent or rimless glasses are hard to segment reliably. SAM may return weak or fragmented masks.
- The current target is full glasses region, not lens-only segmentation.
- VQA is used as a signal, not a strict gate. If grounding boxes are found, the pipeline can continue.
- Very large Florence boxes are filtered out using a fixed area ratio threshold (`MAX_BBOX_AREA_RATIO = 0.28`). This is a design choice to reduce false positives.
- Multi-person handling is not implemented as a separate mode. The pipeline is tuned for single-face style images.
- Thresholds are fixed in code (for example SAM confidence and bbox filter ratio). This is a design choice for a simple baseline.
