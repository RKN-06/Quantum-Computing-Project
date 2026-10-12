# Binary and three-class classification extension

## Scope and status
Team CtrlQ. This is an executed extension, not a replacement for the original binary evidence. Both tasks use the same financial observations, temporal boundaries and explicit 16-feature allowlist. Original comparison snapshots remain under `reports/stage5`; new snapshots are under `reports/multiclass/evidence`.

## Target policy
- Binary: 0 means next observed close is lower/equal, 1 means higher.
- Three-class: 0 DOWN for simple return < -0.0025; 1 NEUTRAL for -0.0025 <= return <= 0.0025; 2 UP for return > 0.0025.
- The neutral band is prespecified before the extension run, not selected from evaluation scores. It is an illustrative modelling convention, not a transaction-cost estimate or economically validated threshold. No claim is made that it is optimal.
- Missing next prices remain missing. Labels use original ticker history before filtering. `SP500_targets.csv` stores forward outcomes and label end dates separately. These columns MUST NOT enter predictors.
- Five-class classification remains a later extension.

## Notebook changes
Every code cell has a preceding Markdown heading and a one-line brief describing what that cell does (reviewed 2026-10-12). Notebook 01 retains its data preparation logic. Notebook 02 preserves the original 19-column binary features export and additionally writes target metadata. Notebooks 03 and 04 keep legacy binary sections and append separately named extension runs. Notebook 05 preserves the original evidence analysis and appends independently verified extension comparisons and charts.

## Temporal protocol
The fixed five stocks are AAPL, MSFT, JPM, XOM and JNJ. Deterministic evenly spaced dates are chosen without looking at outcomes. Training: up to 200 dates in 2015–2020 (1,000 rows). Validation: 80 dates in 2021–2022 (400 rows). Refit: 1,400 development rows. Evaluation: 100 dates in 2023–2024 (500 rows).

All stocks on a sampled date remain in the same partition. Purge labels crossing boundaries. Check every class has at least 20 observations per partition. This check is a basic support safeguard, not a proof of statistical adequacy. Do not tune sample sizes or thresholds based on evaluation scores. The already-inspected evaluation period remains exploratory, with cross-stock and temporal dependence.

## Models and selection
Notebook 03 uses StandardScaler and all 16 features with dummy, linear, RBF and polynomial SVMs. Notebook 04 compares those models with QSVM on identical 2- or 4-component angle-scaled PCA inputs. Fit all transforms using training only for selection, then refit on development. Each task independently selects C in {0.1,1,10} by validation balanced accuracy. RBF also searches gamma in {scale,0.01,0.1}; polynomial degrees are 2 or 3. Ties follow grid order. This is a common protocol, not equal tuning compute.

QSVM uses an exact noiseless ZZ fidelity statevector kernel, one repetition, linear entanglement and classical precomputed-kernel SVC. SVC handles three classes through pairwise training. Binary and three-class models reuse label-independent kernels where inputs and preprocessing are identical. There are no shots, device noise, hardware calls or VQC.

## Metrics and comparability
Report accuracy, balanced accuracy, macro precision, macro recall, macro F1 and per-class confusion/recall. Macro F1 differs from the positive-class F1 in legacy binary tables. No multiclass ROC-AUC is calculated from uncalibrated SVC margins. Compare classical versus QSVM within each task and PCA dimension. Do not interpret raw binary-versus-three-class scores as equal-difficulty performance. Full-feature Stage 3 is contextual, not a controlled kernel-only comparison with reduced Stage 4.

## Measured matched results
Both accuracy columns below are balanced accuracy percentages. Difference is QSVM minus validation-selected classical comparator in percentage points. Comparator selection never uses evaluation results.

| task | dimension | classical_model | classical_balanced_accuracy | qsvm_balanced_accuracy | difference_pp |
|---|---|---|---|---|---|
| binary | 2 | rbf | 49.93 | 48.49 | -1.44 |
| binary | 4 | polynomial | 51.34 | 47.73 | -3.62 |
| three_class | 2 | polynomial | 36.82 | 35.16 | -1.66 |
| three_class | 4 | linear | 34.24 | 30.43 | -3.82 |

### All evaluated configurations
Every planned configuration is reported, on the same 500 evaluation rows (100 dates). BA is balanced accuracy. Stage 3 rows use all 16 standardized features; Stage 4 rows use 2 or 4 angle-scaled PCA components. Values are copied from `results/extension/stage3/metrics.csv` and `results/extension/stage4/metrics.csv`.

| Task | Inputs | Model | Selected params | Validation BA | Eval accuracy | Eval BA | Eval macro-F1 |
|---|---|---|---|---:|---:|---:|---:|
| binary | 16 features | dummy | `{}` | 0.5000 | 0.5640 | 0.5000 | 0.3606 |
| binary | 16 features | linear | C=10 | 0.5224 | 0.4860 | 0.4938 | 0.4860 |
| binary | 16 features | rbf | C=10, gamma=scale | 0.5578 | 0.5060 | 0.5064 | 0.5041 |
| binary | 16 features | polynomial | C=10, degree=2 | 0.5291 | 0.5000 | 0.5109 | 0.4998 |
| binary | 2 PCA | dummy | `{}` | 0.5000 | 0.5640 | 0.5000 | 0.3606 |
| binary | 2 PCA | linear | C=1 | 0.5084 | 0.5660 | 0.5044 | 0.3817 |
| binary | 2 PCA | rbf | C=1, gamma=scale | 0.5350 | 0.4640 | 0.4993 | 0.4388 |
| binary | 2 PCA | polynomial | C=0.1, degree=3 | 0.5300 | 0.4600 | 0.4921 | 0.4402 |
| binary | 2 PCA | qsvm | C=1 | 0.5044 | 0.4900 | 0.4849 | 0.4846 |
| binary | 4 PCA | dummy | `{}` | 0.5000 | 0.5640 | 0.5000 | 0.3606 |
| binary | 4 PCA | linear | C=0.1 | 0.5313 | 0.4760 | 0.4839 | 0.4760 |
| binary | 4 PCA | rbf | C=10, gamma=0.01 | 0.5319 | 0.4660 | 0.4797 | 0.4649 |
| binary | 4 PCA | polynomial | C=10, degree=2 | 0.5387 | 0.4940 | 0.5134 | 0.4902 |
| binary | 4 PCA | qsvm | C=1 | 0.5098 | 0.4720 | 0.4773 | 0.4718 |
| three-class | 16 features | dummy | `{}` | 0.3333 | 0.4500 | 0.3333 | 0.2069 |
| three-class | 16 features | linear | C=0.1 | 0.3815 | 0.2640 | 0.3065 | 0.2603 |
| three-class | 16 features | rbf | C=10, gamma=0.01 | 0.3816 | 0.3040 | 0.3414 | 0.3023 |
| three-class | 16 features | polynomial | C=1, degree=3 | 0.3952 | 0.3020 | 0.3599 | 0.2927 |
| three-class | 2 PCA | dummy | `{}` | 0.3333 | 0.4500 | 0.3333 | 0.2069 |
| three-class | 2 PCA | linear | C=1 | 0.3798 | 0.3100 | 0.3815 | 0.2560 |
| three-class | 2 PCA | rbf | C=10, gamma=scale | 0.3908 | 0.3280 | 0.3597 | 0.3278 |
| three-class | 2 PCA | polynomial | C=10, degree=2 | 0.3922 | 0.2920 | 0.3682 | 0.2730 |
| three-class | 2 PCA | qsvm | C=0.1 | 0.3623 | 0.2880 | 0.3516 | 0.2775 |
| three-class | 4 PCA | dummy | `{}` | 0.3333 | 0.4500 | 0.3333 | 0.2069 |
| three-class | 4 PCA | linear | C=10 | 0.3985 | 0.2700 | 0.3424 | 0.2425 |
| three-class | 4 PCA | rbf | C=10, gamma=0.01 | 0.3817 | 0.2900 | 0.3531 | 0.2731 |
| three-class | 4 PCA | polynomial | C=0.1, degree=2 | 0.3839 | 0.2820 | 0.3429 | 0.2645 |
| three-class | 4 PCA | qsvm | C=10 | 0.3362 | 0.3000 | 0.3043 | 0.2957 |

Polynomial models use gamma=scale. The dummy always predicts the majority training class (UP in both tasks).

### What the extension results show
- In all four task and dimension cells, the QSVM's balanced accuracy is below its validation-selected classical comparator, by 1.44 to 3.82 percentage points.
- Binary: on this larger 500-row evaluation, no model is clearly above the 50% chance level. The best is the 4-component polynomial SVM at 51.34%. Both QSVMs (48.49% and 47.73%) are below the dummy's 50%.
- Three-class: chance balanced accuracy is 33.33%. The 2-component QSVM reaches 35.16%; the 4-component QSVM reaches 30.43%, below the dummy. The highest evaluation score (2-component linear, 38.15%) was not the validation-selected comparator, so it is not used for the comparison.
- Raw accuracy is misleading here. The all-UP dummy reaches 56.4% (binary) and 45.0% (three-class) raw accuracy at exactly chance balanced accuracy. Only one SVM beats it on raw accuracy (binary 2-component linear, 56.6%), and that model correctly labels 98.6% of UP rows but only 2.3% of DOWN rows.
- The binary rerun does not reproduce the above-chance balanced accuracies of the original 100-row Stage 4 run (QSVM 51.65% and 54.14%). The two runs use different samples, so the original results stand as recorded, but they should be read as small-sample estimates.

## Evaluation class support
| task | label | rows | dates |
|---|---|---|---|
| binary | 0 | 218 | 100 |
| binary | 1 | 282 | 100 |
| three_class | 0 | 166 | 100 |
| three_class | 1 | 109 | 100 |
| three_class | 2 | 225 | 100 |

## Resource accounting
Final kernels are 1,400 × 1,400 (development) and 500 × 1,400 (evaluation). Measured final kernel construction took 9.09 s and 3.63 s for 2 qubits, and 9.55 s and 4.12 s for 4 qubits (`results/extension/stage4/kernel_diagnostics.json`). Logical circuit depth is 5 (2 qubits, 2 CX gates) and 11 (4 qubits, 6 CX gates). Training-kernel minimum eigenvalues are within 2e-13 of zero.

Per-model timing records include preprocessing, fit, prediction and QSVM training/evaluation kernel build times. Kernels are actually built once per representation and reused across tasks, so summing per-model attributed costs overstates physical workflow duration. Kernel-array bytes exclude process memory and statevector caches. Logical depth and gate counts are not transpiled hardware costs. Timings are single-machine measurements with no speedup claim.

## Verification and reproducibility
Use Python 3.12 with requirements-stage4.txt for all updated notebooks (the extension records Qiskit versions even in the classical notebook). Run Notebooks 01 through 05 from the project root or notebooks folder. The existing legacy snapshot analysis is self-contained, but the appended extension audit requires outputs from Notebooks 03 and 04.

Assertions check missing labels, boundary classes, feature allowlists, temporal purging, shared evaluation keys and actual labels across configurations, all-class support, training-only preprocessing, kernel symmetry/diagonal/range/PSD and an independent overlap. Notebook 05 recomputes five metrics for all 28 model/task configurations (140 metric checks). `reports/multiclass/evidence/manifest.json` freezes 20 compact evidence files with SHA-256 hashes, and metadata records input hashes, helper-code hash (which matches the current `experiments/multiclass_extension.py`), package versions and configurations. `reports/multiclass/per_class_metrics.csv`, the per-configuration confusion-matrix JSON files and `verification.json` are also saved there, but they are not covered by the manifest hashes.

Generated full feature and target CSVs stay local and are reproducible from notebooks; compact evidence is committed. No user-owned src files are changed.

## Remaining work
**Planned:** sample and qubit scaling, circuit variations (feature map, repetitions, entanglement), larger regime coverage, time-aware statistical uncertainty using date-aware resampling, a fresh confirmatory evaluation period, source-price verification and a survivorship-bias review.

**Deferred:** five-class classification, and shot-based or noise-model simulation.

**Optional, not performed:** hardware execution and backtesting.

No statistical significance, quantum advantage or trading profitability follows from these results.
