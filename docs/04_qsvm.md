# Notebook 04: PCA, quantum encoding and initial QSVM

## Purpose
`notebooks/04_qsvm.ipynb` compares quantum-kernel SVMs with matched dummy, linear, RBF and polynomial baselines. It reports two prespecified PCA dimensions (2 and 4). This is an exact, noiseless statevector experiment, with classical SVM optimization and no hardware calls.

## Inputs and reproduction
Inputs are the cleaned eligible-stock CSV, the feature CSV and `feature_columns.json` under `data/processed/`. Regenerate `SP500_features.csv` with Notebook 02 if needed; the generated feature CSV is not duplicated in this stage commit.

In VS Code select `E:/Quantum Project/.venv-stage4/Scripts/python.exe` as the kernel and Run All in Notebook 04. The isolated environment preserves the Stage 3 installation. On a new machine, create a Python 3.12 virtual environment and install `requirements-stage4.txt`, then select it as the kernel. Each run rewrites local `results/stage4` outputs.

## Frozen experimental design
The stock set is AAPL, MSFT, JPM, XOM and JNJ, matching Stage 3. Date sampling is deterministic and independent of outcomes. Up to 40 training dates, 16 validation dates and 20 evaluation dates are evenly spaced within their periods, with all available selected stocks kept on each date. This differs from Stage 3's larger sample, so do not attribute cross-notebook score differences solely to quantum kernels.

Actual next-observation dates are recovered from source price histories. Labels reaching the next partition are purged. Saved target values are checked against those prices. All stocks share chronological boundaries.

## Actual partitions
| Partition | Rows | Dates | First date | Last date | UP fraction |
|---|---:|---:|---|---|---:|
| train | 200 | 40 | 2015-01-02 | 2020-12-30 | 0.535 |
| validation | 80 | 16 | 2021-01-04 | 2022-12-29 | 0.600 |
| development_refit | 280 | 56 | 2015-01-02 | 2022-12-29 | 0.554 |
| evaluation | 100 | 20 | 2023-01-03 | 2024-12-30 | 0.680 |

## Preprocessing and parameter selection
The pipeline is StandardScaler, full-SVD PCA, then MinMaxScaler mapping to [-pi, pi] with clipping. It is first fitted using training rows only. Validation rows are transformed without fitting. Hyperparameters are selected by validation balanced accuracy; both PCA dimensions are retained rather than selecting a dimension using evaluation scores.

All SVMs use balanced class weights and C in [0.1, 1, 10]. RBF also searches gamma in [scale, 0.01, 0.1]. Polynomial also searches degree in [2, 3] with gamma=scale. The dummy uses the most frequent training class. This matches observations and representations, not search compute budgets. Ties follow declared grid order.

After selection, preprocessing and models are refitted on training plus validation rows. Evaluation data only passes through the fitted transforms. All five models receive the same angle inputs at a given dimension. Fitted preprocessing parameters and clipping fractions are saved.

## Quantum architecture
The feature map is `zz_feature_map`, with one repetition and linear entanglement, on 2 or 4 qubits. `FidelityStatevectorKernel(shots=None)` computes exact squared state overlaps. A classical `SVC(kernel="precomputed")` is fitted on the training kernel and predicts from the evaluation-to-training kernel.

There is no shot noise, physical device noise, mitigation or hardware validation. Circuit depths and two-qubit operation counts are logical circuit resources, not hardware-transpiled resources.

## PCA retention and clipping
| Components | Development variance retained | Evaluation coordinates clipped |
|---|---:|---:|
| 2 | 0.6596 | 0.0000 |
| 4 | 0.8386 | 0.0025 |

Clipping is reported per component value, not per observation. Retained variance is not a measure of predictive information.

## Executed evaluation results
| Components | Model | Balanced accuracy | ROC-AUC | F1 | Average precision |
|---|---|---:|---:|---:|---:|
| 2 | dummy | 0.5000 | 0.5000 | 0.8095 | 0.6800 |
| 2 | linear | 0.5790 | 0.5979 | 0.7259 | 0.7479 |
| 2 | rbf | 0.5377 | 0.5942 | 0.6167 | 0.7478 |
| 2 | polynomial | 0.5836 | 0.5786 | 0.6500 | 0.7185 |
| 2 | qsvm | 0.5165 | 0.5101 | 0.7050 | 0.6727 |
| 4 | dummy | 0.5000 | 0.5000 | 0.8095 | 0.6800 |
| 4 | linear | 0.5588 | 0.5538 | 0.4950 | 0.7301 |
| 4 | rbf | 0.5000 | 0.5386 | 0.0000 | 0.7222 |
| 4 | polynomial | 0.4844 | 0.5528 | 0.3617 | 0.7083 |
| 4 | qsvm | 0.5414 | 0.5358 | 0.5370 | 0.6819 |

Average precision is the precision-recall summary, not trapezoidal PR-AUC. These are descriptive point estimates, with no confidence intervals or significance tests. No model or dimension is promoted on the basis of evaluation scores.

## Runtime and kernel resources
| Qubits | Final training kernel | Evaluation kernel | Training kernel seconds | Evaluation kernel seconds | Logical depth | Two-qubit gates |
|---|---|---|---:|---:|---:|---:|
| 2 | [280, 280] | [100, 280] | 0.4289 | 0.2219 | 5 | 2 |
| 4 | [280, 280] | [100, 280] | 0.5060 | 0.3130 | 11 | 6 |

The notebook separately records preprocessing, model fitting, prediction and kernel-construction times. Quantum kernel construction is excluded from the SVC-only fit timer and must be added for end-to-end comparisons. Kernel array bytes are measured; peak process memory is not. Timing depends on the machine and simulator caching and cannot demonstrate quantum speedup.

## Verification
All code cells executed successfully, and notebook schema validation passed. Checks cover label agreement, chronological label purging, finite input values, both classes in each partition, and training-only fitted preprocessing. Evaluation transforms leave fitted parameters unchanged. Both dimensions use the same observations and target values.

Kernel checks cover matrix dimensions, finite values, symmetry, unit diagonals, [0,1] bounds and positive semidefiniteness within numerical tolerance. Selected entries were independently checked using statevector overlaps. Saved prediction files independently reproduce every reported balanced accuracy and ROC-AUC. Input hashes match Stage 3's input hashes.

## Outputs
The executed notebook embeds parameter choices, metrics, split counts, kernel diagnostics, PCA retention, clipping charts, matched comparison plots, kernel heatmaps and QSVM confusion matrices. Local `results/stage4` also stores exact sample keys, predictions, all validation trials, preprocessing parameters, package versions and SHA-256 input hashes. Generated output files remain ignored by Git.

## Limitations and remaining work
The small five-stock sample and earlier long-history eligibility can introduce survivorship bias. Price adjustments remain unverified, and next-observation labels may span missing market sessions. A single small validation interval can yield unstable hyperparameters. Financial observations are correlated by market date and ticker.

The 2023–2024 period was previously inspected in this project and is an exploratory evaluation, not a fresh final holdout. No claims of quantum advantage, statistical significance, generalization to all index constituents, or financial profitability are supported. Statevector simulation on 2–4 qubits is classically tractable.

Planned later stages cover larger matched comparisons, sample/qubit scaling, circuit ablations, uncertainty estimation and the final report. Shot and noise studies are deferred, and hardware execution is optional and not performed. A confirmatory study needs a frozen protocol and an unexamined evaluation period.

## References
- [IBM: ZZ feature map](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.zz_feature_map)
- [Qiskit Machine Learning: FidelityStatevectorKernel](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityStatevectorKernel.html)


## Binary plus three-class update

The original results above document the legacy binary experiment. The updated notebooks retain that evidence and add the protocol documented in [multiclass extension](multiclass_extension.md). Every code cell now has a heading and one-line explanation. Use requirements-stage4.txt for the updated notebook sequence. Three-class output is separate from binary output; five-class and noise remain deferred. The original matched comparison is complete, not future work.
