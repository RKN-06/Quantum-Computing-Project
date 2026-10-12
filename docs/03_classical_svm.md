# Notebook 03: Classical SVM benchmark

## Scope
Executed initial benchmark for AAPL, MSFT, JPM, XOM and JNJ using all 16 engineered features. This is a bounded five-stock demonstration, not a representative evaluation across the whole index. This stage contains no PCA, quantum kernel, noise or hardware experiments.

## Inputs and reproduction
- `data/processed/SP500_features.csv`
- `data/processed/feature_columns.json`
- `data/processed/SP500_cleaned_eligible_tickers.csv`

Generate the feature CSV with Notebook 02 if it is absent; it is not duplicated in this stage's commit. Keep source data accessible through Git LFS.

The pre-existing `.venv` pointed to an unavailable Python installation. An isolated `.venv-stage3` was created for this run. In VS Code select `.venv-stage3/Scripts/python.exe` as the notebook kernel, open `notebooks/03_classical_svm.ipynb`, and Run All. For another machine, create a Python 3.12 environment and install `requirements-stage3.txt`, then select that environment. Do not reuse a copied virtual environment from a different machine.

## Protocol
Development starts in 2015 and ends before 2023. Evaluation uses 2023–2024. Sampling selects at most 500 evenly spaced development dates and 200 evaluation dates, retaining all available selected tickers at each date without examining labels. Date boundaries are shared across stocks.

Three expanding development folds use a one-sampled-date gap. Each label's actual next-observation date is recovered from the price history. Training rows whose labels reach validation are removed. Development labels reaching the test boundary and evaluation labels reaching 2025 are excluded. Sampling never recomputes targets.

StandardScaler is inside each estimator pipeline and fitted separately for each fold. Selection uses balanced accuracy only on development folds. C values are 0.1, 1 and 10; RBF gamma values are scale, 0.01 and 0.1; polynomial degrees are 2 and 3 with gamma=scale. SVMs use balanced class weights. The dummy predicts the majority training class.

## Actual split sizes
- development: 2500 rows, 5 stocks, 2015-01-02 through 2022-12-29.
- evaluation: 1000 rows, 5 stocks, 2023-01-03 through 2024-12-30.

## Executed results
| Model | Accuracy | Balanced accuracy | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| dummy | 0.4730 | 0.5000 | 0.0000 | 0.5000 | 0.5270 |
| linear | 0.5040 | 0.5105 | 0.4537 | 0.5204 | 0.5283 |
| rbf | 0.5170 | 0.5198 | 0.5056 | 0.5227 | 0.5295 |
| polynomial | 0.4850 | 0.4830 | 0.5155 | 0.4789 | 0.5062 |

Metrics are descriptive. Average precision is used as the precision-recall summary; it is not trapezoidal PR-AUC. Search time includes all candidates and final refit, while prediction time covers labels and scores.

## Verification and outputs
All code cells executed successfully. Assertions verified finite features, unique ticker-date keys, binary class availability, agreement with source-price targets, and temporal separation including label end dates. Notebook schema validation passed.

The executed notebook embeds split summaries, model parameters, metrics and charts. Local `results/stage3` contains metrics, per-row predictions, all CV scores, fold boundaries, split counts, environment versions, input SHA-256 hashes and three chart files. These generated outputs remain ignored by Git; the notebook contains the reviewable results.

## Interpretation and limitations
Compare the SVM metrics against the dummy rather than judging raw accuracy in isolation. No statistical significance or financial profitability has been demonstrated. Financial rows are correlated within dates and tickers.

The 2023–2024 period was previously examined in project experiments and must be described as exploratory rather than an untouched confirmatory holdout. Do not change hyperparameters after viewing these results. A future final evaluation needs a separately frozen protocol and an unexamined period.

The five named stocks and the earlier long-history eligibility screen can introduce survivorship bias. The price-adjustment policy remains unverified. Next-observation labels can span missing trading sessions. The earlier HUBB price imputation is not source-verified (HUBB is not selected here).

## Next stage
Fit dimensionality reduction and bounded quantum mappings on development training data. Compare quantum and classical kernels on exactly matched observations and representations. Advanced scaling, noise, hardware and statistical studies remain future stages.
