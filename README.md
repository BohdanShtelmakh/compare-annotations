# Annotation comparator

This app compares annotated objects with ground-truth objects.

## Logic

- JSON objects are loaded and normalized by `nm`.
- Geometry comes from a PNG mask when available, otherwise from a polygon path or bounding box.
- Objects are matched only when their types match. IDs (`nm`) are not used for matching.
- One-to-one matching maximizes total intersection area. Bounding boxes are used to skip impossible mask comparisons.
- Each match reports IoU. Missing objects and extra annotations have zero intersection and contribute their own area to the union.
- The overall score is `sum(intersection) / sum(union) * 100`.

## Usage

Install dependencies and run:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

`main.py` reads the sample files under `data/`, prints the comparison, and writes the full result to `output/results.json`. The `output/` directory is ignored by Git.

Run tests with:

```bash
.venv/bin/pytest -q
```
