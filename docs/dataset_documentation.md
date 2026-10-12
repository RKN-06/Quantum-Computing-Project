# Dataset Documentation

## Overview

This project uses historical US equity-market data for evaluating classical Support Vector Machines and Quantum Support Vector Machines.

All CSV files are stored with Git LFS because they are too large for standard GitHub file storage.

## Files

| File | Role | Status |
|---|---|---|
| `data/raw/SP500_Historical_Data.csv` | Primary raw source dataset | Main dataset for this project |
| `data/processed/SP500_cleaned_eligible_tickers.csv` | Cleaned modelling dataset | Generated in Notebook 01 |
| `data/market_raw.csv` | Supporting market dataset | Retained for future review and possible experiments |
| `data/market_aligned.csv` | Supporting aligned market dataset | Retained for future review and possible experiments |

## Primary source dataset

**File:** `data/raw/SP500_Historical_Data.csv`

The dataset contains daily historical market data for 472 tickers.

### Coverage

- Date range: 2000-01-03 to 2026-02-20
- Rows: 2,703,531
- Tickers: 472

### Columns

| Column | Description |
|---|---|
| `Ticker` | Stock ticker symbol |
| `Date` | Trading date |
| `Open` | Opening price |
| `High` | Highest daily price |
| `Low` | Lowest daily price |
| `Close` | Closing price |
| `Adj Close` | Adjusted closing price |
| `Volume` | Daily trading volume |

## Quality checks

Notebook `01_data_understanding.ipynb` checks for:

- Missing values
- Duplicate ticker-date records
- Invalid or non-positive prices
- Negative volume
- Daily High and Low inconsistencies
- Chronological ordering within each ticker

One inconsistent Low value was found for HUBB on 2021-05-05. The notebook corrects it only in the working dataframe; the original raw dataset remains unchanged.

## Cleaned modelling dataset

**File:** `data/processed/SP500_cleaned_eligible_tickers.csv`

The cleaned dataset contains only stocks with at least 5,000 trading days of history.

- Eligible tickers: 372
- Rows: 2,405,879
- Date range: 2000-01-03 to 2026-02-20

This is the input for later feature-engineering and machine-learning notebooks.

## Limitations

- The dataset contains 472 tickers rather than exactly 500.
- Historical index membership must be reviewed before making broad claims about S&P 500 representation.
- The constituent selection may introduce survivorship bias if it reflects a modern ticker list.
- The datasets are historical research inputs and must not be interpreted as investment advice.

## Binary plus three-class update

The original results above document the legacy binary experiment. The updated notebooks retain that evidence and add the protocol documented in [multiclass extension](multiclass_extension.md). Every code cell now has a heading and one-line explanation. Use requirements-stage4.txt for the updated notebook sequence. Three-class output is separate from binary output; five-class and noise remain deferred. The original matched comparison is complete, not future work.

Notebook 02 additionally generates `data/processed/SP500_targets.csv` with identifiers, binary/three-class labels, simple forward returns and label-end dates. This is target metadata, never model input. It is ignored by Git and reproducible; the 16-feature allowlist and original feature CSV schema remain unchanged.
