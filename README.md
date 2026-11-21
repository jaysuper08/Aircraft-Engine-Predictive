# Aircraft Engine Predictive Maintenance
CS655 final project — Victor Superales — October 31, 2026.

Compares Linear Regression, Random Forest and LSTM for NASA C-MAPSS FD001
remaining useful life (RUL), with current-only/history ablations. The report
contains results measured by running the accompanying code; no scores are invented.
This is a simulated-data benchmark, not a validated aircraft maintenance system.

## Quick start (Python 3.12)
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
python -m src.visualize
python scripts/verify_results.py
python -m src.predict --input data/raw/test_FD001.txt --model RF_history --output results/predictions_new.csv
python scripts/build_report.py
```
The report builder requires `pdflatex` (TeX Live/MiKTeX). Training uses CPU and
two threads. Expect runtime to depend on CPU and library versions.
The delivered `report/CS655_Aircraft_Engine_Project.pdf` can be read without rerunning.
Raw FD001 files are included; `src.data.download()` retrieves them from the archive
linked by NASA if absent. Do not overwrite the delivered results without making a copy.
`requirements-lock.txt` records exact versions used; `requirements.txt` allows
compatible versions, which may change numeric outputs. A CPU PyTorch build can be
installed using its official CPU wheel index if a default install is too large.

## Reproducibility protocol
- 80 engine IDs for training, 20 different engine IDs for validation, NumPy split seed 42.
- Original NASA 100-engine test file is never used for model selection or scaling.
- Training-only variance filtering and standardization; sensor IDs, split IDs,
  censoring cuts, data SHA-256 hashes and runtime versions are in `results/manifest.json`.
- Target = min(remaining cycles,125). Report gives both capped-target and original
  uncapped-target scores. Predictions are clipped to [0,125].
- Training windows: causal trailing 30 cycles, stride 2. Short histories repeat
  the earliest available reading on the left. No future readings are used.
- One fixed pseudo-censored endpoint per validation engine; one last observed
  endpoint per test engine. Validation cuts are fractions of complete held-out
  lifetimes to mimic censoring; this is an imperfect censoring model.
- Two forest configurations per feature set and two LSTM configurations are
  selected only on seed-42 validation endpoint RMSE. LSTM uses early stopping.
- Fixed selected configurations are repeated for RF-history and LSTM with seeds
  42,43,44. Validation still selects the checkpoint within each LSTM run.
- Models are kept trained on the 80 training engines; no final 100-engine refit.
- Main comparisons use predesignated seed 42, never the best test seed.
- Bootstrap intervals resample the 100 independent test-engine endpoints, not
  overlapping sensor windows. They are conditional on a fitted model and do not
  capture every source of uncertainty.

## Contents
- `src/data.py`: data access, target generation, causal windows, feature summaries.
- `src/models.py`: PyTorch LSTM and fit/predict wrapper.
- `src/train.py`: preprocessing, tuning, repeated training, evaluation, persistence.
- `src/metrics.py`: metrics and engine-endpoint bootstrap.
- `src/visualize.py`: figures generated from actual data and results.
- `src/predict.py`: inference with saved models.
- `tests/test_pipeline.py`: checks labels, data shape, causality, padding, features and split separation.
- `results/metrics.csv`, `predictions.csv`: all scores and per-engine predictions.
- `results/tuning.json`: validation candidates and selected hyperparameters.
- `results/models/`: fitted preprocessing, forests, linear models, LSTM weights.
- `report/`: PDF and editable LaTeX source.
- `scripts/build_report.py`: regenerates tables and report from measured results.

## Publish to GitHub
This folder is prepared for a repository; it has not been published to an account.
Create an empty repository on GitHub, then from this folder:
```bash
git init
git add .
git commit -m "Add reproducible aircraft engine RUL project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
Replace the URL with your own repository URL. The course requires a hosted repository;
a ZIP alone does not fulfill that part of submission.

## Sources and attribution
NASA data: Saxena & Goebel (2008), Turbofan Engine Degradation Simulation Data Set.
Original archive README is in `data/raw/readme.txt`; provenance URL is recorded in
`src/data.py`. The publisher's README has a column-label typo at its end; the
26 numeric columns are engine, cycle, 3 settings, and 21 sensors.
See report references for the original simulation paper, Random Forest, LSTM,
Adam, NumPy, pandas, scikit-learn, PyTorch and Matplotlib. Libraries implement
standard algorithms; the project-specific data, experiment and evaluation code
was prepared with AI assistance. Review the code, rerun it, and follow your
professor's rules for acknowledging assistance before submission.
