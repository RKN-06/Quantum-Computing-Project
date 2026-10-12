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
Every code cell has a preceding Markdown heading and one-line explanation. Notebook 01 retains its data preparation logic. Notebook 02 preserves the original 19-column binary features export and additionally writes target metadata. Notebooks 03 and 04 keep legacy binary sections and append separately named extension runs. Notebook 05 preserves the original evidence analysis and appends independently verified extension comparisons and charts.

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

## Evaluation class support
| task | label | rows | dates |
|---|---|---|---|
| binary | 0 | 218 | 100 |
| binary | 1 | 282 | 100 |
| three_class | 0 | 166 | 100 |
| three_class | 1 | 109 | 100 |
| three_class | 2 | 225 | 100 |

## Resource accounting
Per-model timing records include preprocessing, fit, prediction and QSVM training/evaluation kernel build times. Kernels are actually built once per representation and reused across tasks, so summing per-model attributed costs overstates physical workflow duration. Kernel-array bytes exclude process memory and statevector caches. Logical depth and gate counts are not transpiled hardware costs. Timings are single-machine measurements with no speedup claim.

## Verification and reproducibility
Use Python 3.12 with requirements-stage4.txt for all updated notebooks (the extension records Qiskit versions even in the classical notebook). Run Notebooks 01 through 05 from the project root or notebooks folder. The existing legacy snapshot analysis is self-contained, but the appended extension audit requires outputs from Notebooks 03 and 04.

Assertions check missing labels, boundary classes, feature allowlists, temporal purging, shared evaluation keys and actual labels across configurations, all-class support, training-only preprocessing, kernel symmetry/diagonal/range/PSD and an independent overlap. Notebook 05 recomputes five metrics for all 28 model/task configurations (140 metric checks). SHA-256 manifests freeze compact evidence, and metadata records input hashes, helper-code hash, package versions and configurations. Per-class metrics and confusion matrices are also committed.

Generated full feature and target CSVs stay local and are reproducible from notebooks; compact evidence is committed. No user-owned src files are changed.

## Remaining work
Larger regime coverage, statistical uncertainty using date-aware resampling, source-price verification, survivorship-bias review, scaling and circuit variations remain open. Five-class/noise experiments are deferred. Optional hardware/backtesting require separate implementation. No significance, general quantum advantage or trading profitability follows from these results.
