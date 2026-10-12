# QSVM vs Classical SVM on Financial Data

Team **CtrlQ**. This project compares classical support vector machines with simulated quantum-kernel SVMs (QSVMs) for predicting next-day stock direction. It is a benchmarking study and makes no claim of quantum advantage or trading profitability.

## Status

| Part | Status |
|---|---|
| Original binary comparison | Complete |
| Binary plus three-class extension | Executed, exploratory |
| Scaling and statistical validation | Planned |
| Five-class classification and noise simulation | Deferred |
| Hardware execution and backtesting | Optional, not performed |

In every matched comparison so far, the QSVM scores below its validation-selected classical comparator. See `documentation.md` for the full results.

## Setup

Use Python 3.12. The raw data file `data/raw/SP500_Historical_Data.csv` is stored with Git LFS, so install Git LFS before cloning.

```bash
git lfs install
git clone https://github.com/RKN-06/Quantum-Computing-Project.git
cd Quantum-Computing-Project
python -m venv .venv-stage4
.venv-stage4/Scripts/python -m pip install -r requirements-stage4.txt
```

On macOS or Linux, use `.venv-stage4/bin/python` instead.

## Running

Select `.venv-stage4` as the Jupyter kernel and run the notebooks in order:

1. `notebooks/01_data_understanding.ipynb`: cleans the source data and screens eligible tickers.
2. `notebooks/02_feature_engineering.ipynb`: builds the 16 features and the binary and three-class targets.
3. `notebooks/03_classical_svm.ipynb`: trains the classical SVM baselines on all 16 features.
4. `notebooks/04_qsvm.ipynb`: runs the matched classical and QSVM comparison on 2 or 4 PCA components.
5. `notebooks/05_comparison_analysis.ipynb`: recomputes the metrics from saved predictions and analyzes cost.

Notebook 05 needs the outputs from notebooks 03 and 04. Shared extension code is in `experiments/multiclass_extension.py`.

## Outputs

- `reports/stage5/`: frozen evidence from the original binary comparison.
- `reports/multiclass/`: binary and three-class extension results, plots and SHA-256-hashed evidence.
- `results/` and the generated CSVs in `data/processed/` are recreated by the notebooks and are not committed.

## Documentation

- `documentation.md`: project design, current status and results.
- `docs/01_data_understanding.md` to `docs/05_comparison_analysis.md`: one write-up per notebook.
- `docs/multiclass_extension.md`: the binary and three-class protocol and full results.
- `docs/dataset_documentation.md`: data source and limitations.

## Limitations

The 2023–2024 evaluation period had already been examined, so the results are exploratory. All QSVM results come from exact, noiseless statevector simulation on 2 to 4 qubits. The sample covers five stocks, and no confidence intervals have been computed yet.
