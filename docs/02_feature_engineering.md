# Notebook 02: Financial Feature Engineering

## Purpose

`notebooks/02_feature_engineering.ipynb` converts cleaned historical stock data into numerical features and a next-observation direction target for classical SVM and QSVM experiments.

## Input

**File:** `data/processed/SP500_cleaned_eligible_tickers.csv`

The input contains 2,405,879 rows across 372 stocks selected for having at least 5,000 historical observations. Dates are parsed as datetime values, and records are sorted by `Ticker` and `Date`.

## Price-column choice

The comparison found no material numerical differences between `Close` and `Adj Close` using `np.isclose`.

Features and targets provisionally use `Close`. Equality between these columns does not establish whether the prices account for splits and dividends. The dataset provider's adjustment policy remains to be verified.

## Engineered features

Calculations are performed separately for each ticker.

| Feature group | Columns | Calculation |
|---|---|---|
| Log returns | `return_1d`, `return_5d`, `return_10d`, `return_20d` | Natural logarithm of current close divided by the close n observations earlier |
| Rolling volatility | `volatility_5d`, `volatility_10d`, `volatility_20d` | Rolling sample standard deviation of one-observation log returns |
| Moving-average distance | `ma_distance_5d`, `ma_distance_10d`, `ma_distance_20d`, `ma_distance_50d` | Current close divided by the rolling mean close, minus one |
| Intraday range | `intraday_range` | `(High - Low) / Close` |
| Open-to-close return | `close_open_return` | `log(Close / Open)` |
| Volume change | `volume_log_change` | Difference between consecutive log-volume values |
| Relative volume | `volume_zscore_20d` | Current volume minus its rolling mean, divided by its rolling standard deviation |
| Momentum | `rsi_14d` | RSI calculated from simple rolling averages of gains and losses |

**Total: 16 input features.**

The `d` suffix denotes trading observations. Exact trading-calendar continuity has not yet been verified.

## RSI convention

RSI uses 14-observation simple averages rather than Wilder's smoothing:

`RSI = 100 * average_gain / (average_gain + average_loss)`

A completely flat window receives a neutral value of 50. Incomplete windows remain missing.

## Prediction target

The next closing price is obtained with `groupby("Ticker")["Close"].shift(-1)`.

The forward log return is `log(next_close / current_close)`.

The label `target_up` is:

- **1:** the forward return is positive.
- **0:** the forward return is zero or negative.

The final observation for each ticker has no known next price and receives a missing target.

The prediction is made after the current session closes. Where records are missing, the target refers to the next available observation rather than necessarily the next trading session.

## Missing and invalid values

- Rolling features require complete windows.
- Zero volume is treated as unavailable when calculating log volume.
- Zero rolling volume standard deviation produces a missing z-score.
- Infinite feature values are converted to missing values.
- Rows with missing features or unknown targets are removed.
- Returns and targets are not recalculated after removing rows.

A separate dataframe, `feature_df`, holds the resulting dataset.

## Validation performed

The final working dataset passed checks for:

1. No missing input features.
2. No infinite input features.
3. Targets restricted to 0 and 1.
4. No duplicate ticker-date records.

These outcomes were reported from the user's notebook execution. These checks verify structural consistency; they do not independently establish price accuracy or absence of all forms of bias.

## Resulting dataset

| Measure | Result |
|---|---:|
| Input rows | 2,405,879 |
| Removed rows | 18,954 |
| Remaining rows | 2,386,925 |
| DOWN or unchanged | 1,171,449 (49.08%) |
| UP | 1,215,476 (50.92%) |

Class proportions describe the complete feature dataset. Training and evaluation splits will be assessed separately.

## Outputs

### Feature dataset

`data/processed/SP500_features.csv`

Contains 19 columns: `Ticker`, `Date`, the 16 input features, and `target_up`. The intermediate `forward_return_1d` column is excluded from this export.

### Feature list

`data/processed/feature_columns.json`

Stores the explicit list of 16 model inputs. Model training must select these columns rather than automatically using every numerical column.

## Leakage controls and limitations

Features use current and historical observations within each ticker. Future prices are used only for target construction.

Scaling, PCA, feature selection, and model fitting have not been performed in this notebook. They must be fitted using training data only in the subsequent modelling stage.

Remaining limitations include:

- Unverified price-adjustment policy.
- Possible missing trading sessions.
- Possible survivorship bias in the original ticker universe.
- Eligibility based on total available history, which may use information unavailable at an earlier prediction date.
- The inherited HUBB Low-value replacement is an imputation, not a source-verified historical correction.

## Next stage

`03_classical_svm.ipynb` will define chronological splits and a manageable experiment sample, apply training-only preprocessing, and evaluate dummy, linear, RBF, and polynomial SVM baselines.

For pooled multi-stock experiments, split boundaries must be based on dates across all tickers. Training labels that extend into a validation or test period must be excluded.


## Binary plus three-class update

The original results above document the legacy binary experiment. The updated notebooks retain that evidence and add the protocol documented in [multiclass extension](multiclass_extension.md). Every code cell now has a heading and one-line explanation. Use requirements-stage4.txt for the updated notebook sequence. Three-class output is separate from binary output; five-class and noise remain deferred. The original matched comparison is complete, not future work.

Notebook 02 additionally generates `data/processed/SP500_targets.csv` with identifiers, binary/three-class labels, simple forward returns and label-end dates. This is target metadata, never model input. It is ignored by Git and reproducible; the 16-feature allowlist and original feature CSV schema remain unchanged.
