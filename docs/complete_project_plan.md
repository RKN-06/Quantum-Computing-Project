# Complete benchmark and IBM hardware plan

## What is now fixed

`experiments/complete_benchmark.py` is the clean, code-first benchmark. It
rebuilds a frozen 20-stock feature table from the cleaned OHLCV file, uses adjusted
close for return and target construction, keeps the target metadata in memory,
and creates four chronological partitions:

- train: 2015-2020
- validation: 2021-2022
- primary test: 2023-2024
- previously inspected robustness period: 2025 onward

The primary comparison includes a dummy baseline, five classical models, and
four exact statevector quantum-kernel SVM variants. All models use exactly the
same stock-date rows in every partition. The training, validation, test and robustness
sets currently contain 1,599, 800, 1,200 and 1,200 rows respectively. Kernel
construction grows quadratically with sample count. Hyperparameters use only
validation balanced accuracy. The 2025 onward period has already been inspected
and does not qualify as an untouched confirmatory test.

The run saves per-row predictions and immediately audits every reported metric,
identical matched evaluation keys and actual labels, and 10-date moving-block
bootstrap intervals for each QML model versus the classical model selected on
validation. The notebook rejects stale or incomplete saved results. These
intervals describe uncertainty for this fixed exploratory sample; they do not
establish a population-wide or quantum-advantage claim.

Run:

```powershell
.venv-stage4\Scripts\python.exe experiments\complete_benchmark.py --task both
```

Results are written under `results/complete_benchmark/`.

## IBM Quantum Open Plan

IBM's Open Plan provides limited free QPU time. Create an IBM Quantum account,
create an instance, and generate an API key and instance CRN in the IBM Quantum
Platform. Install the added runtime dependency, then save credentials locally:

```powershell
.venv-stage4\Scripts\python.exe -m pip install qiskit-ibm-runtime==0.47.0
.venv-stage4\Scripts\python.exe -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(token='YOUR_API_KEY', instance='YOUR_INSTANCE_CRN', overwrite=True)"
```

Run the small hardware audit:

```powershell
.venv-stage4\Scripts\python.exe experiments\ibm_hardware_audit.py --shots 512
```

The audit submits only four 4-qubit compute-uncompute circuits. It records the
backend, job ID, finite-shot fidelity, and ideal statevector fidelity. It is a
hardware validation experiment, not a claim that the full QSVM ran on hardware.
Never commit the API key or saved credential file.
