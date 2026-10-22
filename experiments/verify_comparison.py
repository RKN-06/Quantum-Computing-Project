"""Audit saved benchmark predictions and estimate date-block uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def metrics(actual, prediction):
    return {"accuracy": accuracy_score(actual, prediction),
            "balanced_accuracy": balanced_accuracy_score(actual, prediction),
            "macro_f1": f1_score(actual, prediction, average="macro", zero_division=0)}


def date_block_interval(left: pd.DataFrame, right: pd.DataFrame,
                        block_dates: int = 10, repeats: int = 1000) -> dict:
    """Paired moving-block bootstrap over dates; all stocks on a date stay together."""
    joined = left.merge(right, on=["Ticker", "Date", "actual"],
                        suffixes=("_qml", "_classical"), validate="one_to_one")
    assert len(joined) == len(left) == len(right)
    dates = np.sort(joined.Date.unique())
    if len(dates) < block_dates:
        raise ValueError("Not enough dates for block bootstrap")
    groups = [joined.index[joined.Date.eq(date)].to_numpy() for date in dates]
    rng = np.random.default_rng(42)
    estimates = []
    blocks_needed = int(np.ceil(len(dates) / block_dates))
    for _ in range(repeats):
        starts = rng.integers(0, len(dates) - block_dates + 1, size=blocks_needed)
        sampled = np.concatenate([np.arange(s, s + block_dates) for s in starts])[:len(dates)]
        indices = np.concatenate([groups[i] for i in sampled])
        part = joined.iloc[indices]
        if part.actual.nunique() != joined.actual.nunique():
            continue
        estimates.append(100 * (balanced_accuracy_score(part.actual, part.prediction_qml)
                                - balanced_accuracy_score(part.actual, part.prediction_classical)))
    if len(estimates) < repeats * 0.8:
        raise ValueError("Too many bootstrap replicates lost a class")
    point = 100 * (balanced_accuracy_score(joined.actual, joined.prediction_qml)
                   - balanced_accuracy_score(joined.actual, joined.prediction_classical))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"difference_pp": float(point), "ci95_low_pp": float(low),
            "ci95_high_pp": float(high), "valid_replicates": len(estimates),
            "block_dates": block_dates}


def verify_task(output: Path, task: str) -> dict:
    table = pd.read_csv(output / f"{task}_metrics.csv")
    pred = pd.read_csv(output / f"{task}_predictions.csv", parse_dates=["Date"])
    meta = json.loads((output / f"{task}_metadata.json").read_text(encoding="utf-8"))
    expected_qml = {"qsvm_z", "qsvm_zz_linear", "qsvm_pauli_linear", "qsvm_pauli_full"}
    expected_matched = {"matched_dummy", "matched_logistic", "matched_linear_svm",
                        "matched_rbf_svm", "matched_random_forest", "matched_hist_gradient_boosting"}
    assert len(meta["tickers"]) == 20 and len(set(meta["tickers"])) == 20
    assert set(table.model) == expected_qml | expected_matched
    assert set(table.model) == set(pred.model) == set(meta["models"])
    checks = 0
    for row in table.itertuples(index=False):
        for period, prefix in [("test", "test"), ("robustness", "confirmatory")]:
            part = pred.loc[(pred.model == row.model) & (pred.period == period)]
            assert not part.duplicated(["Ticker", "Date"]).any()
            split_name = "confirmatory" if period == "robustness" else "test"
            assert len(part) == meta["splits"][split_name]["rows"]
            for name, value in metrics(part.actual, part.prediction).items():
                assert np.isclose(value, getattr(row, f"{prefix}_{name}"), atol=1e-12), (row.model, period, name)
                checks += 1
    for period in ("test", "robustness"):
        reference = pred.loc[(pred.model == "qsvm_zz_linear") & (pred.period == period),
                             ["Ticker", "Date", "actual"]].sort_values(["Date", "Ticker"]).reset_index(drop=True)
        for model in expected_qml | expected_matched:
            part = pred.loc[(pred.model == model) & (pred.period == period),
                            ["Ticker", "Date", "actual"]].sort_values(["Date", "Ticker"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(part, reference)
    test_reference = pred.loc[(pred.model == "qsvm_zz_linear") & (pred.period == "test"),
                              ["Ticker", "Date", "actual"]]
    matched = table.loc[table.model.isin(expected_matched - {"matched_dummy"})]
    comparator = matched.loc[matched.validation_balanced_accuracy.idxmax(), "model"]
    classical = pred.loc[(pred.model == comparator) & (pred.period == "test")]
    comparisons = {}
    for model in sorted(expected_qml):
        qml = pred.loc[(pred.model == model) & (pred.period == "test")]
        comparisons[model] = date_block_interval(qml, classical)
    report = {"task": task, "verified_metric_values": checks,
              "matched_test_rows": len(test_reference), "matched_test_dates": test_reference.Date.nunique(),
              "validation_selected_classical": comparator,
              "paired_date_block_comparisons": comparisons,
              "status": "exploratory; 2025 onward was previously inspected"}
    (output / f"{task}_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "three_class", "both"], default="both")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "results" / "complete_benchmark"
    for task in (["binary", "three_class"] if args.task == "both" else [args.task]):
        print(json.dumps(verify_task(output, task), indent=2))
