# Performance Evaluation of Quantum Support Vector Machines (QSVM) vs Classical SVMs on High-Dimensional Financial Data

## 1. Project overview

This project evaluates whether quantum-kernel Support Vector Machines (QSVMs) provide useful classification performance, robustness, or modelling characteristics relative to carefully tuned classical SVMs on financial data with many engineered features. It is a benchmarking and reproducibility project—not a claim that quantum machine learning will outperform classical methods.

The primary task is directional classification: use information available at a market close to predict whether the next chosen return horizon is positive or non-positive. The work compares predictive quality, computational cost, sensitivity to feature dimension, and resilience to realistic quantum noise.

## 2. Motivation and problem statement

Financial prediction data is noisy, non-stationary, and prone to accidental look-ahead bias. Kernel methods are attractive because they can express nonlinear relationships, but their computational cost can grow sharply with sample size. Quantum feature maps offer a different implicit feature space, yet their practical benefit on realistic financial tasks remains uncertain.

**Problem statement.** Design a leakage-safe, time-aware experimental framework that compares classical SVM kernels with quantum-kernel SVMs on a common financial classification task. Determine, with uncertainty reported, how the approaches differ across predictive performance, resource use, scaling, circuit choices, and simulated noise.

## 3. Objectives and research questions

### Objectives

1. Build a reproducible financial-data and feature-engineering pipeline.
2. Establish strong classical SVM baselines before assessing quantum methods.
3. Evaluate QSVM kernels on reduced, bounded feature representations.
4. Measure accuracy **and** computational and quantum-resource costs.
5. Test sensitivity to dimensionality, circuit design, shots, and noise.
6. Optionally validate a deliberately small experiment on IBM Quantum hardware.

### Research questions

- How do QSVM and classical linear, RBF, and polynomial SVMs compare on strictly out-of-sample temporal data?
- Does a quantum feature map add value beyond classical nonlinear kernels at equivalent input dimension?
- How do accuracy, AUC, F1, and calibration vary with the number of retained features and training observations?
- How do circuit depth, entanglement, shots, and noise affect the quantum kernel?
- What are the runtime, kernel-construction, memory, and circuit-resource trade-offs?

## 4. Scope and success criteria

**In scope:** binary daily (or other explicitly fixed horizon) direction classification; public market data; engineered technical, return, volatility, volume, and cross-asset features; simulator-first QSVM experiments; time-series validation; transparent reporting.

**Out of scope unless separately added:** live trading, investment advice, causal claims, high-frequency execution, and claims of quantum advantage. A negative QSVM result is a valid outcome.

Success means the experiment is reproducible, leakage-safe, benchmarked fairly, and reports uncertainty and constraints. It does not require QSVM to win.

## 5. Prerequisites and software stack

| Area | Recommended tools |
|---|---|
| Language/environment | Python 3.10+; `venv` or Conda; Jupyter for exploration |
| Data and computation | pandas, NumPy, SciPy, yfinance or a documented alternative |
| Classical ML | scikit-learn, joblib |
| Quantum ML | Qiskit, qiskit-aer, qiskit-machine-learning, qiskit-algorithms |
| Experiment tracking | MLflow or structured JSON/CSV plus Git commit hashes |
| Visualisation | matplotlib, seaborn, optionally plotly |
| Testing/quality | pytest, ruff/black, pre-commit |

Pin dependency versions in `requirements.txt` or `environment.yml`. Record random seeds, backend/noise-model versions, device configuration, data retrieval date, and all hyperparameters.

## 6. Dataset strategy

Start with a small, defensible universe: one liquid equity index ETF (for example, SPY) or a basket of large liquid equities. Use adjusted OHLCV data, document provider and retrieval time, and cache the raw download unchanged. A broader later stage may add sector ETFs, index constituents, macro proxies, and a panel formulation.

Choose a period long enough to include multiple regimes (calm, stress, recovery) where data licensing permits. Never mix adjusted and unadjusted fields without documenting the treatment. Missing dates, splits, holidays, duplicated rows, and implausible prices must be validated before features are created.

### Target definition

For each decision date `t`, define the forward log return for horizon `h`:

`r(t, h) = log(close[t + h] / close[t])`

The primary label is `y(t) = 1` if `r(t,h) > 0`, otherwise `0`. Fix `h` (initially one trading day) before evaluation. Optional robustness targets include a neutral band, e.g. positive only when `r(t,h) > τ`, with neutral observations removed or modelled as a third class. The operational timing must be explicit: features through close `t` can only support a decision after that close, for a next-session outcome.

## 7. End-to-end pipeline

```text
Raw OHLCV / auxiliary series
  → schema and quality validation
  → chronological alignment and adjusted-price policy
  → feature engineering using only past/current observations
  → forward target creation and unavailable-label removal
  → chronological train / validation / final-test split
  → fit transforms on training windows only
  → feature selection or PCA within each training fold
  → tune classical SVM baselines with walk-forward CV
  → choose bounded reduced input for quantum experiments
  → construct classical / quantum kernel matrices
  → evaluate held-out periods and record resources
  → ablations, noise, scaling, and statistical comparison
  → report results, limitations, and reproducible artifacts
```

## 8. Financial feature engineering

Every feature at date `t` must be computable using information available no later than `t`.

- **Returns and momentum:** 1/2/5/10/20-day log returns, rolling cumulative returns, moving-average spread, RSI, MACD-derived values.
- **Volatility and risk:** rolling standard deviation, ATR, downside volatility, rolling beta to a benchmark, drawdown, realized-volatility proxies.
- **Volume and liquidity:** volume change, rolling volume z-score, volume/average-volume, price-volume trend.
- **Price structure:** intraday range, close-to-open return, Bollinger-band position, distance from rolling highs/lows.
- **Cross-market/context (optional):** lagged returns/volatility of sector ETFs, broad index, bond, FX, or volatility proxies—after trading-calendar alignment.
- **Regime/context (optional):** rolling trend and volatility regime indicators. Do not use future regime labels.

Use a documented warm-up period; discard rows whose rolling inputs are incomplete. Retain feature definitions in a data dictionary. Initial quantum runs should use 2–8 selected or PCA-compressed features because each input dimension generally requires a qubit or an encoding scheme.

## 9. Leakage prevention and temporal validation

Leakage controls are non-negotiable:

- Sort and split by timestamp before fitting scalers, imputers, selectors, PCA, or models.
- Fit each transform only on its relevant training fold, then apply it to validation/test data.
- Shift any indicator if its computation assumes unavailable end-of-day information at decision time.
- Purge overlapping-label observations around split boundaries for multi-day horizons; add an embargo where appropriate.
- Do not randomly shuffle rows or use ordinary IID cross-validation.
- Keep the final test interval untouched until all model and circuit choices are frozen.

Recommended structure: an early training period, a chronological validation/development period used via expanding-window or rolling-window `TimeSeriesSplit`, and a final held-out test period. For each fold, log its exact dates and number of observations. A walk-forward evaluation may additionally refit at a specified cadence to emulate deployment.

## 10. Classical SVM baselines

Build baselines first, using the same features, folds, and tuning budget as practical.

1. Dummy classifier and majority-class accuracy.
2. Linear SVC / linear-kernel SVC.
3. RBF SVC, tuned over `C` and `gamma`.
4. Polynomial SVC, tuned conservatively over degree, `C`, `gamma`, and `coef0`.
5. Optional non-SVM reference: logistic regression and gradient boosting, clearly labelled as contextual baselines.

Use a pipeline containing imputation (if justified), robust/standard scaling, reduction/selection, and classifier so fit boundaries are enforced. Account for class imbalance using class weights and report balanced metrics; do not select a model using final-test results.

## 11. Dimensionality reduction and input preparation

High-dimensional raw features are useful for the classical benchmark, but QSVM input must be deliberately constrained. Compare: (a) filter selection based only on training data, (b) model-based selection, and (c) PCA retaining a fixed component count. Scale first where the technique requires it.

For angle encodings, map each retained feature into a bounded interval such as `[-π, π]` using training-set scaling/clipping rules. Persist the mapping. Compare the classical SVM on both the full engineered feature set and the same reduced representation used by QSVM; this separates quantum-kernel effects from dimensionality-reduction effects.

## 12. Quantum kernel / QSVM architecture

In this project, QSVM means a classical SVM trained using a kernel matrix estimated from a quantum feature map—not a claim that optimization itself runs on a quantum computer.

1. Reduce/scale the input to `d` bounded values.
2. Encode each vector `x` into a parameterized circuit `Uφ(x)` on approximately `d` qubits.
3. Estimate similarity, typically `K(x,z) = |<φ(x)|φ(z)>|²`, with a statevector simulator, shot-based simulator, or hardware execution.
4. Build training and test kernel matrices and train `SVC(kernel='precomputed')`.

Candidate feature maps include `ZZFeatureMap`, Pauli feature maps, and custom shallow maps with single-qubit rotations plus configurable entanglement. Start with 2–4 qubits and shallow repetitions. Confirm the resulting kernel is numerically usable (finite, symmetric within tolerance, sensible diagonal, and positive-semidefinite handling documented).

## 13. Evaluation and resource metrics

### Predictive metrics

Report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC (where scores are available), PR-AUC for imbalance, confusion matrix, and per-fold values. Include majority-class and naive directional baselines. If probability calibration is used, perform it inside temporal validation and report Brier score/calibration plots.

### Computational metrics

Measure wall-clock time separately for preprocessing, model tuning, kernel construction, fitting, and prediction. Record peak memory where feasible, sample counts, feature/qubit count, backend, shots, circuit depth, two-qubit gate count, and kernel-matrix dimensions. Kernel methods commonly require O(n²) storage and pairwise evaluation; present this as a central limitation, not an afterthought.

## 14. Experiment programme

| Experiment family | Controlled variations | Primary outputs |
|---|---|---|
| Baseline benchmark | linear/RBF/poly; full vs reduced inputs | predictive metrics, runtime |
| Dimension scaling | 2, 4, 6, 8 retained dimensions/qubits where feasible | accuracy vs dimension; cost curve |
| Sample scaling | increasing chronological training sizes | runtime/memory/kernel-build scaling |
| Circuit ablation | feature map, repetitions, entanglement topology | metric/resource comparison |
| Shot study | statevector and selected finite shot counts | variance and cost trade-off |
| Noise study | depolarizing, readout, and backend-inspired noise | degradation versus ideal simulation |
| Robustness | alternate asset, period, horizon, threshold | stability of conclusions |
| Optional hardware | tiny frozen best/representative configuration | simulator–hardware gap |

Keep the experiment matrix manageable. Freeze a small set of configurations based on development folds before the final test. Each run should have a unique ID and machine-readable configuration file.

## 15. Noise and optional IBM Quantum hardware validation

Use ideal statevector simulation to establish a reference, then use finite-shot simulation and documented noise models. Vary one dimension at a time when possible: readout error, one-/two-qubit depolarization, shot count, or circuit depth. Report mean and variability over repeated kernel estimates/seeds.

Hardware validation is optional and should be small: few qubits, limited samples, frozen circuit and preprocessing, and a predeclared runtime budget. Record backend name, calibration timestamp, transpilation settings, shots, queue/execution details, and mitigation method if used. Never compare a hardware result to an ideal simulator without identifying the mismatch. Hardware evidence validates feasibility for that tiny setup; it does not establish broad practical advantage.

## 16. Statistical validation and financial interpretation

Report fold distributions, mean/median and spread, not only a best score. Use paired, time-aware comparisons between models on identical out-of-sample predictions. Suitable options include block bootstrap confidence intervals, a corrected resampled comparison, or a Diebold–Mariano-style forecast comparison where assumptions are carefully stated. Avoid treating correlated daily observations as independent.

An optional illustrative backtest can translate predicted directions into a simple long/cash or long/short rule with a lag consistent with the label timing. Include transaction-cost assumptions, turnover, cumulative return, volatility, Sharpe-like ratio, maximum drawdown, and benchmark comparison. It is an interpretation layer—not evidence of a tradable strategy—and must not drive model selection on the final test set.

## 17. Expected visualisations

- Chronological data/split timeline and class balance by period.
- Feature correlation/importance view based on training data only.
- Walk-forward metric distributions and ROC/PR curves for held-out predictions.
- Confusion matrices for final held-out evaluation.
- Accuracy/F1/AUC against retained dimension and training sample size.
- Kernel-construction runtime and memory against sample size.
- Circuit depth/two-qubit-gate count against performance.
- Noise-strength and shot-count sensitivity plots.
- Optional cumulative-return and drawdown charts, clearly labelled illustrative.

## 18. Suggested repository structure

```text
project-root/
├── README.md
├── QSVM_FINANCIAL_DATA_PROJECT_DOCUMENTATION.md
├── requirements.txt
├── configs/                 # versioned YAML/JSON experiment settings
├── data/
│   ├── raw/                 # immutable cached downloads (normally gitignored)
│   ├── interim/
│   └── processed/
├── notebooks/               # exploration only; promote stable logic to src/
├── src/
│   ├── data/                # acquisition, validation, target creation
│   ├── features/            # feature definitions and transforms
│   ├── validation/          # temporal splits, purge/embargo logic
│   ├── models/              # classical and quantum kernels/models
│   ├── experiments/         # runners and resource measurement
│   └── visualization/
├── tests/
├── results/                 # metrics, plots, tables; keep raw runs traceable
└── reports/                 # final report/slides
```

## 19. Delivery plan

### Overnight 50% milestone

- Create the environment, repository skeleton, configuration convention, and reproducibility metadata.
- Acquire/cache one instrument's adjusted OHLCV data and add validation checks.
- Implement target creation, a documented initial feature set, and chronological split logic.
- Implement leakage-safe preprocessing and linear/RBF/poly SVM baselines.
- Produce an initial classical benchmark table and split/feature diagnostic plots—labelled preliminary, without claims about QSVM.
- Implement a minimal simulator quantum-kernel proof of plumbing on a small reduced dataset.

### Later 50% roadmap

- Complete hyperparameter tuning with walk-forward validation and freeze the protocol.
- Expand QSVM configurations, circuit/shot/dimension ablations, and resource logging.
- Run noise studies, sample-scaling experiments, and robustness checks.
- Optionally run the predeclared small hardware validation.
- Apply statistical comparisons, assemble figures/tables, and write the final report and reproducibility guide.

## 20. Risks and limitations

| Risk | Mitigation |
|---|---|
| Look-ahead leakage | temporal pipelines, audited timestamps, untouched final test |
| Non-stationarity | walk-forward evaluation and regime/period robustness checks |
| Class imbalance | balanced metrics, class weights, baseline comparison |
| Kernel cost | small controlled samples, scaling curves, explicit resource reporting |
| Noisy/deep circuits | shallow maps, simulator noise study, limited hardware claim |
| Over-tuning | predeclared matrix, development/final-test separation, configuration logging |
| Data revisions/licensing | cached raw snapshot, provider/date documentation |

## 21. Expected outcomes, résumé value, and deliverables

Expected outcomes are evidence, not predetermined performance: a fair comparison may show a classical kernel is superior, a QSVM is competitive only in narrow settings, or a quantum feature map has interesting but costly behaviour. The report should identify where conclusions are reliable and where scale, noise, or limited samples prevent stronger claims.

The résumé value is strongest when described honestly: *Built a reproducible, leakage-aware benchmark comparing classical SVM kernels and Qiskit quantum-kernel QSVMs for temporal financial classification; evaluated accuracy, scaling, circuit design, and noise robustness.* Add hardware validation only if it was actually completed and documented.

Final deliverables:

1. Version-controlled source code, configurations, tests, and dependency lockfile.
2. Cached-data instructions and a data dictionary (subject to provider terms).
3. Reproducible experiment tables, figures, and resource logs.
4. A professional report documenting methods, results, uncertainty, limitations, and no unsupported claims.
5. Optional notebook/demo and presentation summarising the final evidence.

