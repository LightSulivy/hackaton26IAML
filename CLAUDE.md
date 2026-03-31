# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML hackathon — binary classification to predict patient mortality status in 2019 (MORTSTAT_2019), evaluated on **F1-score**.

## Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
.venv/bin/jupyter notebook baseline.ipynb
```

The Jupyter kernel is registered as `hackathon` (Python 3.10).

## Data (in `data/`, git-ignored)

| File | Description |
|---|---|
| `data.csv` | 59 065 rows × ~1 525 features (148 MB) — SEQN is the patient ID |
| `ground_truth_train.csv` | SEQN + MORTSTAT_2019 (binary) for ~54 065 patients |
| `test_indexes.csv` | 5 000 SEQN to predict |
| `features_metadata.csv` | Feature descriptions: Component, function, pathology |

## Architecture

`baseline.ipynb` is the single entry point:
1. Load all four CSVs
2. Split train/test by SEQN membership
3. Drop columns with >80% missing values
4. Train LightGBM with 5-fold stratified CV (early stopping)
5. Generate submission CSV

## Submission format

File name: `[idgroupe]_[idsoumission].csv`
Content: 5 000 rows, 2 columns (SEQN, prediction), sorted by SEQN ascending, **no header**.

```python
submission.to_csv(filename, index=False, header=False)
```

Upload to: https://utbox.univ-tours.fr/s/mS8549YmdjmX4Tk

## Key constraints

- Optimize for **F1-score**, not accuracy — dataset is class-imbalanced
- LightGBM handles NaN natively — no mandatory imputation step
- Test set has exactly 5 000 patients — validate `len(submission) == 5000` before submitting
