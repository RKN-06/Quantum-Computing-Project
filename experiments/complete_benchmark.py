"""Chronological exploratory benchmark for classical models and simulator QSVMs.

The script builds the 20-stock modelling table from the cleaned
OHLCV file and derives targets from the next observed adjusted close. This
avoids depending on the ignored, generated SP500_targets.csv artifact.

It compares five classical models and four exact statevector QSVM variants. A dummy is reported as
the baseline. Model selection uses 2015-2020 training data and 2021-2022
validation data only. 2023-2024 is the primary test and 2025 onward is a
previously inspected robustness period that is never used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


# Fixed 20-stock universe: four times the original five-stock panel.
# The universe is frozen before model fitting and is not selected by results.
TICKERS = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "AMZN", "GOOGL", "NVDA",
    "BAC", "WMT", "PG", "KO", "CVX", "HD", "DIS", "PFE", "MRK", "ABT", "ADBE", "AMD",
]
FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "ma_distance_5d", "ma_distance_10d", "ma_distance_20d",
    "ma_distance_50d", "intraday_range", "close_open_return",
    "volume_log_change", "volume_zscore_20d", "rsi_14d",
]
THRESHOLD = 0.0025


def make_table(root: Path) -> pd.DataFrame:
    """Build causal features and binary/three-class targets for 20 stocks."""
    path = root / "data" / "processed" / "SP500_cleaned_eligible_tickers.csv"
    use = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = pd.read_csv(path, usecols=use, parse_dates=["Date"])
    df = df.loc[df.Ticker.isin(TICKERS)].sort_values(["Ticker", "Date"]).copy()
    g = df.groupby("Ticker", sort=False)
    price = df["Adj Close"].where(df["Adj Close"] > 0, df["Close"])
    log_price = np.log(price)
    one = log_price.groupby(df.Ticker).diff()
    df["return_1d"] = one
    for w in (5, 10, 20):
        df[f"return_{w}d"] = log_price - log_price.groupby(df.Ticker).shift(w)
        df[f"volatility_{w}d"] = one.groupby(df.Ticker).transform(
            lambda s, w=w: s.rolling(w, min_periods=w).std()
        )
    for w in (5, 10, 20, 50):
        ma = price.groupby(df.Ticker).transform(
            lambda s, w=w: s.rolling(w, min_periods=w).mean()
        )
        df[f"ma_distance_{w}d"] = price / ma - 1
    # Intraday ratios use one consistent price basis. Adjustment cancels in
    # (adjusted high - adjusted low) / adjusted close.
    df["intraday_range"] = (df.High - df.Low) / df.Close
    df["close_open_return"] = np.log(df.Close / df.Open)
    log_volume = np.log(df.Volume.where(df.Volume > 0))
    df["volume_log_change"] = log_volume.groupby(df.Ticker).diff()
    mean = g.Volume.transform(lambda s: s.rolling(20, min_periods=20).mean())
    std = g.Volume.transform(lambda s: s.rolling(20, min_periods=20).std())
    df["volume_zscore_20d"] = (df.Volume - mean) / std.replace(0, np.nan)

    def rsi(s: pd.Series) -> pd.Series:
        change = s.diff()
        gain = change.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-change.clip(upper=0)).rolling(14, min_periods=14).mean()
        total = gain + loss
        return (100 * gain / total.replace(0, np.nan)).mask(total == 0, 50)

    df["rsi_14d"] = price.groupby(df.Ticker).transform(rsi)
    next_price = price.groupby(df.Ticker).shift(-1)
    simple_return = next_price / price - 1
    df["target_up"] = (simple_return > 0).astype("Int64")
    df["target_three"] = pd.Series(
        np.select([simple_return < -THRESHOLD, simple_return > THRESHOLD], [0, 2], default=1),
        index=df.index, dtype="Int64"
    ).mask(simple_return.isna())
    df["label_end"] = df.groupby("Ticker").Date.shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["label_end"])
    df = df.loc[df[FEATURES].notna().all(axis=1)].copy()
    df["target_up"] = df.target_up.astype(int)
    df["target_three"] = df.target_three.astype(int)
    return df[["Ticker", "Date", "label_end", *FEATURES, "target_up", "target_three"]]


def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bounds = {
        "train": ("2015-01-01", "2021-01-01"),
        "validation": ("2021-01-01", "2023-01-01"),
        "test": ("2023-01-01", "2025-01-01"),
        "confirmatory": ("2025-01-01", "2026-03-01"),
    }
    result = {}
    for name, (lo, hi) in bounds.items():
        part = df.loc[(df.Date >= lo) & (df.Date < hi) & (df.label_end < hi)].copy()
        if part.empty:
            raise ValueError(f"No rows in {name}; check dataset coverage.")
        result[name] = part.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    if result["train"].label_end.max() >= result["validation"].Date.min():
        raise AssertionError("Training labels cross the validation boundary")
    if result["validation"].label_end.max() >= result["test"].Date.min():
        raise AssertionError("Validation labels cross the test boundary")
    if result["test"].label_end.max() >= result["confirmatory"].Date.min():
        raise AssertionError("Test labels cross the confirmatory boundary")
    return result


def transforms() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=4, svd_solver="full")),
        ("angles", MinMaxScaler(feature_range=(-np.pi, np.pi), clip=True)),
    ])


def candidates() -> dict[str, tuple[object, list[dict]]]:
    return {
        "dummy": (DummyClassifier(strategy="most_frequent"), [{}]),
        "logistic": (LogisticRegression(class_weight="balanced", max_iter=3000),
                     [{"C": c} for c in (0.01, 0.1, 1, 10)]),
        "linear_svm": (SVC(kernel="linear", class_weight="balanced"),
                       [{"C": c} for c in (0.01, 0.1, 1, 10)]),
        "rbf_svm": (SVC(kernel="rbf", class_weight="balanced"),
                    [{"C": c, "gamma": g} for c in (0.1, 1, 10) for g in ("scale", 0.01, 0.1)]),
        "random_forest": (RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
                          [{"n_estimators": n, "max_depth": d} for n in (200, 500) for d in (3, 8, None)]),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(random_state=42),
                                    [{"max_iter": n, "learning_rate": lr, "max_leaf_nodes": leaf}
                                     for n in (100, 200) for lr in (0.03, 0.1) for leaf in (7, 15)]),
    }


def score(y, pred) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def record_predictions(records: list, frame: pd.DataFrame, task: str,
                       model: str, period: str, target: str, pred) -> None:
    part = frame[["Ticker", "Date", target]].copy()
    part.columns = ["Ticker", "Date", "actual"]
    part["task"] = task
    part["model"] = model
    part["period"] = period
    part["prediction"] = pred
    records.append(part)


def run(root: Path, task: str = "binary") -> tuple[pd.DataFrame, dict]:
    print(f"[{task}] loading and preparing data", flush=True)
    df = make_table(root)
    population = split_data(df)
    frames = sampled_frames(population)
    predictions = []
    print(f"[{task}] running matched classical models", flush=True)
    rows = run_matched_classical(frames, task, predictions)
    print(f"[{task}] matched classical models complete", flush=True)
    rows.extend(run_qsvm(root, frames, task, predictions))
    out = pd.DataFrame(rows).sort_values("test_balanced_accuracy", ascending=False)
    output = root / "results" / "complete_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    out.to_csv(output / f"{task}_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / f"{task}_predictions.csv", index=False)
    metadata = {"task": task, "models": out.model.tolist(), "features": FEATURES,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "tickers": TICKERS, "scale_factor_vs_original": 4, "threshold": THRESHOLD,
                "splits": {k: {"rows": len(v), "dates": v.Date.nunique(),
                               "start": str(v.Date.min().date()), "end": str(v.Date.max().date())}
                           for k, v in frames.items()},
                "sampling": "all models use identical sampled stock-date rows in every partition",
                "python": platform.python_version(),
                "source_sha256": hashlib.sha256((root / "data/processed/SP500_cleaned_eligible_tickers.csv").read_bytes()).hexdigest()}
    (output / f"{task}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    verifier = runpy.run_path(str(root / "experiments" / "verify_comparison.py"))
    verifier["verify_task"](output, task)
    return out, metadata


def sampled_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    limits = {"train": 80, "validation": 40, "test": 60, "confirmatory": 60}
    sampled = {}
    for name, frame in frames.items():
        dates = np.sort(frame.Date.unique())
        chosen = dates[np.linspace(0, len(dates) - 1, min(limits[name], len(dates)), dtype=int)]
        sampled[name] = frame.loc[frame.Date.isin(chosen)].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return sampled


def run_matched_classical(frames: dict[str, pd.DataFrame], task: str, predictions: list) -> list[dict]:
    """Classical controls on exactly the rows and transforms used by QML."""
    small = frames
    target = "target_up" if task == "binary" else "target_three"
    pre = transforms()
    A = pre.fit_transform(small["train"][FEATURES])
    V = pre.transform(small["validation"][FEATURES])
    dev = pd.concat([small["train"], small["validation"]], ignore_index=True)
    final_pre = transforms()
    D = final_pre.fit_transform(dev[FEATURES])
    T = final_pre.transform(small["test"][FEATURES])
    C = final_pre.transform(small["confirmatory"][FEATURES])
    rows = []
    for name, (template, grid) in candidates().items():
        best = None
        for params in grid:
            model = template.__class__(**template.get_params()).set_params(**params)
            model.fit(A, small["train"][target])
            ba = balanced_accuracy_score(small["validation"][target], model.predict(V))
            if best is None or ba > best[0]:
                best = (ba, params)
        model = template.__class__(**template.get_params()).set_params(**best[1])
        started = time.perf_counter()
        model.fit(D, dev[target])
        elapsed = time.perf_counter() - started
        test_pred = model.predict(T)
        robust_pred = model.predict(C)
        record_predictions(predictions, small["test"], task, "matched_" + name, "test", target, test_pred)
        record_predictions(predictions, small["confirmatory"], task, "matched_" + name, "robustness", target, robust_pred)
        row = {"task": task, "model": "matched_" + name,
               "parameters": json.dumps(best[1], sort_keys=True),
               "validation_balanced_accuracy": float(best[0]),
               "fit_seconds": elapsed, "qsvm_train_rows": len(dev),
               "qsvm_kernel_note": "same sampled dates and preprocessing as QML"}
        row.update({f"test_{k}": v for k, v in score(small["test"][target], test_pred).items()})
        row.update({f"confirmatory_{k}": v for k, v in score(small["confirmatory"][target], robust_pred).items()})
        rows.append(row)
    return rows


def run_qsvm(root: Path, frames: dict[str, pd.DataFrame], task: str, predictions: list) -> list[dict]:
    from qiskit.circuit.library import pauli_feature_map, z_feature_map, zz_feature_map
    from qiskit.quantum_info import Statevector

    def states(circuit, values):
        return np.asarray([
            Statevector.from_instruction(circuit.assign_parameters(row)).data
            for row in values
        ], dtype=np.complex128)

    def fidelity(left, right):
        # Exact statevector fidelity, batched in native BLAS rather than
        # constructing a Python/Qiskit circuit for every pair of rows.
        return np.abs(left @ right.conj().T) ** 2

    target = "target_up" if task == "binary" else "target_three"
    small = frames
    pre = transforms()
    A = pre.fit_transform(small["train"][FEATURES])
    V = pre.transform(small["validation"][FEATURES])
    y_train = small["train"][target]
    y_valid = small["validation"][target]
    feature_maps = {
        "qsvm_z": z_feature_map(4, reps=1),
        "qsvm_zz_linear": zz_feature_map(4, reps=1, entanglement="linear"),
        "qsvm_pauli_linear": pauli_feature_map(4, reps=1, paulis=["X", "ZZ"], entanglement="linear"),
        "qsvm_pauli_full": pauli_feature_map(4, reps=1, paulis=["Z", "ZZ"], entanglement="full"),
    }
    dev = pd.concat([small["train"], small["validation"]], ignore_index=True)
    X_dev = pre.fit_transform(dev[FEATURES])
    X_test = pre.transform(small["test"][FEATURES])
    X_confirm = pre.transform(small["confirmatory"][FEATURES])
    rows = []
    for name, feature_map in feature_maps.items():
        print(f"[{task}] {name}: computing exact statevectors and kernels", flush=True)
        kernel_started = time.perf_counter()
        train_states = states(feature_map, A)
        validation_states = states(feature_map, V)
        dev_states = states(feature_map, X_dev)
        test_states = states(feature_map, X_test)
        robustness_states = states(feature_map, X_confirm)
        K = fidelity(train_states, train_states)
        KV = fidelity(validation_states, train_states)
        best = None
        for c in (0.1, 1.0, 10.0):
            model = SVC(kernel="precomputed", C=c, class_weight="balanced", decision_function_shape="ovr")
            model.fit(K, y_train)
            validation = balanced_accuracy_score(y_valid, model.predict(KV))
            if best is None or validation > best[0]:
                best = (validation, c)
        K_dev = fidelity(dev_states, dev_states)
        K_test = fidelity(test_states, dev_states)
        K_confirm = fidelity(robustness_states, dev_states)
        model = SVC(kernel="precomputed", C=best[1], class_weight="balanced", decision_function_shape="ovr")
        model.fit(K_dev, dev[target])
        kernel_seconds = time.perf_counter() - kernel_started
        test_pred = model.predict(K_test)
        robust_pred = model.predict(K_confirm)
        record_predictions(predictions, small["test"], task, name, "test", target, test_pred)
        record_predictions(predictions, small["confirmatory"], task, name, "robustness", target, robust_pred)
        row = {"task": task, "model": name, "parameters": json.dumps({"C": best[1]}),
               "validation_balanced_accuracy": float(best[0]), "fit_seconds": kernel_seconds,
               "qsvm_train_rows": len(dev), "qsvm_kernel_note": "exact batched statevector fidelity; 80/40/60/60 dates",
               "qsvm_feature_map": name.replace("qsvm_", "")}
        row.update({f"test_{k}": v for k, v in score(small["test"][target], test_pred).items()})
        row.update({f"confirmatory_{k}": v for k, v in score(small["confirmatory"][target], robust_pred).items()})
        rows.append(row)
        print(f"[{task}] {name}: complete in {kernel_seconds:.1f} s", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "three_class", "both"], default="both")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = ["binary", "three_class"] if args.task == "both" else [args.task]
    for task_name in tasks:
        table, _ = run(root, task_name)
        print(f"\n{task_name}\n", table.to_string(index=False))
