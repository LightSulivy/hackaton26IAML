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
| `data.csv` | 59 064 rows × 1 540 columns — SEQN is the patient ID |
| `ground_truth_train.csv` | SEQN + MORTSTAT_2019 (binary) for 54 064 patients |
| `test_indexes.csv` | 5 000 SEQN to predict |
| `features_metadata.csv` | 1 539 features — Component (Lab/Exam/Questionnaire/Demo), function, pathology |

## Architecture

`baseline.ipynb` is the single entry point:
1. Load all four CSVs
2. Split train/test by SEQN membership (54 064 train / 5 000 test)
3. Drop columns with >80% missing — removes 927 cols → 612 features remain
4. Train LightGBM with 5-fold stratified CV (early stopping patience=50)
5. Retrain final model on full train set
6. Generate submission CSV

## Baseline results (LightGBM, default params)

| Param | Value |
|---|---|
| `n_estimators` | 500 |
| `learning_rate` | 0.05 |
| `num_leaves` | 63 |
| `min_child_samples` | 20 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

**5-fold CV F1 : 0.6945 ± 0.0135**  
Folds: 0.6931 / 0.6951 / 0.7061 / 0.7079 / 0.6702

Test predictions: 4 379 class 0 (alive), 621 class 1 (deceased).

## Submission format

File name: `[idgroupe]_[idsoumission].csv`  
Group: **G1** — increment `SUBMISSION_ID` for each new submission.

Content: 5 000 rows, 2 columns (SEQN, prediction), sorted by SEQN ascending, **no header**.

```python
submission.to_csv(filename, index=False, header=False)
```

Upload to: https://utbox.univ-tours.fr/s/mS8549YmdjmX4Tk

## Key constraints

- Optimize for **F1-score**, not accuracy — class imbalance (~11% deceased)
- LightGBM handles NaN natively — no mandatory imputation step
- Test set has exactly 5 000 patients — validate `len(submission) == 5000` before submitting

## Advanced notebook (`advanced.ipynb`)

Ensemble LightGBM + CatBoost + Random Forest avec threshold optimal sur OOF.

**Améliorations vs baseline :**
- Missing threshold à 70% (vs 80%) — plus de features conservées
- Features NaN : `feat_nb_missing_total/labo/exam/quest/pct` ajoutées par patient
- `features_metadata.csv` : colonne variable = `SAS` (pas `Variable Name`)
- Class imbalance : `scale_pos_weight` (LGB), `class_weights` (CatBoost), `balanced` (RF)
- **Optuna** : 50 trials sur LightGBM (learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda)
- **Poids d'ensemble optimisés** par Nelder-Mead sur OOF
- **Stacking** : meta-learner LogisticRegression sur OOF des 3 modèles
- Threshold optimisé sur OOF (recherche sur [0.20, 0.70])
- Choix automatique entre averaging et stacking selon F1 OOF

Soumission générée : `G1_2.csv`

## AutoGluon notebook (`autogluon.ipynb`)

Exploration automatique de tous les modèles + stacking multi-niveaux, optimisé directement sur F1.

- `presets='best_quality'` + `num_bag_folds=5` + `num_stack_levels=2`
- `TIME_LIMIT=3600` (1h) — augmenter pour de meilleurs résultats
- Modèles sauvegardés dans `autogluon_models/` (git-ignoré)
- Soumission : `G1_3.csv`
- Rechargement sans réentraîner : `TabularPredictor.load('autogluon_models')`

## Dépendances

`requirements.txt` inclut : `lightgbm`, `catboost`, `optuna`, `scikit-learn`, `imbalanced-learn`, `autogluon.tabular`
