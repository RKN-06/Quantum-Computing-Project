# Notebook 01: Dataset Understanding and Preparation

## Purpose

This notebook prepares the raw S&P 500 historical dataset for later financial-machine-learning experiments. It validates the data, identifies quality issues, selects stocks with sufficiently long histories, and saves a cleaned modelling dataset.

## Input dataset

**File:** `data/raw/SP500_Historical_Data.csv`

**Columns:**

- `Ticker`
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close`
- `Volume`

The raw source file is preserved without modification.

## Data-quality checks

The notebook performs the following checks:

1. Loads the CSV and converts `Date` into a datetime field.
2. Confirms the number of rows, columns, unique tickers, and date range.
3. Checks for missing values.
4. Checks for duplicate `Ticker` and `Date` combinations.
5. Checks for invalid prices, negative volume, invalid High values, and invalid Low values.
6. Sorts the data by `Ticker` and `Date`.

## Identified correction

One invalid Low value was found:

- Ticker: `HUBB`
- Date: `2021-05-05`
- Open: `181.34`
- Reported Low: `181.55`

Because a daily Low cannot exceed the Open price, the Low value was corrected to `181.34` in the notebook’s working dataframe only. The raw CSV remains unchanged.

## Stock eligibility rule

The dataset contains stocks with unequal history lengths. To support fair temporal training and testing, the notebook keeps only stocks with at least 5,000 trading days of data.

## Resulting modelling dataset

The cleaned modelling dataset contains:

- 372 eligible stocks
- 2,405,879 rows
- Date range: 2000-01-03 to 2026-02-20

**Output file:** `data/processed/SP500_cleaned_eligible_tickers.csv`

## Next stage

The next notebook, `02_feature_engineering.ipynb`, will create causal financial features and a next-day UP/DOWN target for each eligible stock.