# Data Description - DQN Stock Trading Agent

## 1. Data Source

**File**: `custom-mlp-backprop-cs/dataset/apple_5yr_one.csv`  
**Description**: AAPL daily OHLCV (Open, High, Low, Close, Volume) data spanning approximately 5 years (June 2020 – June 2025)  
**Rows after cleaning**: 1,255 data points (original CSV has 1,257 lines including header and ticker row)  
**Source**: [Kaggle - 2020-2025 Apple Stock Dataset](https://www.kaggle.com/datasets/samanfatima7/2020-2025-apple-stock-dataset)

---

## 2. Raw CSV Format

| Row | Content | Handling |
|-----|---------|----------|
| 1 | `Date,Close,High,Low,Open,Volume` | Header (skipped) |
| 2 | `,AAPL,AAPL,AAPL,AAPL,AAPL` | Ticker row (skipped during loading) |
| 3+ | `2020-06-04,78.329...,79.131...,77.955...,78.832...,87560400` | Data rows (loaded) |

**Loader behavior** (`CsvStockLoader.cs`):
- Reads all lines via `File.ReadAllLines()`
- Skips line 0 (header) and line 1 (ticker row)
- Parses from line 2 onwards (index 2)
- Validates date format (`yyyy-MM-dd`) and numeric columns
- Sorts by date to ensure chronological order

---

## 3. Raw Columns

| Column | Type | Description |
|--------|------|-------------|
| Date | `DateTime` | Trading date (YYYY-MM-DD) |
| Close | `double` | Final stock price at market close |
| High | `double` | Highest price reached during the trading session |
| Low | `double` | Lowest price recorded during the session |
| Open | `double` | Stock price at the start of the trading session |
| Volume | `double` | Total number of shares traded on that day |

**Note**: The CSV column order is `Date,Close,High,Low,Open,Volume` but `StockDataPoint` record expects `Date,Open,High,Low,Close,Volume`. The loader reorders during parsing.

---

## 4. Feature Engineering

Computed by `FeatureEngineer.ComputeFeatures()` — produces **6 features per timestep**:

| Index | Feature | Description | Computation |
|-------|---------|-------------|-------------|
| 0 | **SMA-5** | 5-day Simple Moving Average of Close | Expanding window: averages all available closes up to current day (min 1 day, max 5) |
| 1 | **SMA-20** | 20-day Simple Moving Average of Close | Expanding window: averages all available closes up to current day (min 1 day, max 20) |
| 2 | **Volatility-20** | 20-day rolling standard deviation of daily returns | Daily return = `(Close[t] - Close[t-1]) / Close[t-1]`; rolling std dev (sample, N-1) over up to 20 returns; requires ≥2 valid returns |
| 3 | **Close** | Raw close price (normalized later) | Direct from data |
| 4 | **Cash%** | Normalized cash percentage in portfolio | Placeholder (0.0), filled at runtime by `StockEnvironment` as `Portfolio.Cash / InitialCash`, clamped to [0, 2] |
| 5 | **Holdings%** | Normalized holdings percentage in portfolio | Placeholder (0.0), filled at runtime as `(Portfolio.Holdings × CurrentPrice) / InitialCash`, clamped to [0, 2] |

**NaN handling**: Initial rows where SMA/Volatility have insufficient lookback data produce `NaN`. These are **forward-filled** with the first valid value per feature column (see `ForwardFillNaN()`).

---

## 5. Normalization

**Method**: Min-Max scaling to **[0, 1]** range  
**Class**: `MinMaxNormalizer`  
**Strategy**: **Fit on TRAIN split only (80%), transform both train and test** — prevents data leakage.

### Formula
```
normalized = (value - min) / (max - min)
```

### Implementation Details
- Computes per-feature min/max from training data only
- Ignores `NaN` values during min/max computation
- Handles constant features (min == max) by expanding range slightly (±1e-8) → maps all values to 0.5
- Clamps output to [0, 1] to handle numerical edge cases
- Preserves `NaN` in output (though forward-fill should eliminate them)

---

## 6. Train/Test Split

**Method**: Chronological split — **no random shuffling** (preserves temporal order, prevents leakage)  
**Class**: `TrainTestSplitter.Split()`  
**Ratio**: 80% train / 20% test (configurable, default 0.8)

### Split Sizes
| Split | Steps | Indices |
|-------|-------|---------|
| **Train** | 1,004 | 0 – 1003 |
| **Test** | 251 | 1004 – 1254 |
| **Total** | 1,255 | 0 – 1254 |

### Calculation
```
trainCount = Floor(1255 × 0.8) = 1004
testCount = 1255 - 1004 = 251
```

Both feature arrays and corresponding close price arrays are split identically to maintain alignment.

---

## 7. Environment Parameters

Used when constructing `StockEnvironment` (see `Program.cs` and `StockEnvironment.cs`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| **TransactionFee** | 0.001 (0.1%) | Applied on every buy/sell execution as percentage of trade value |
| **RiskAversion** | 1.0 | Coefficient for risk penalty in reward calculation (drawdown × volatility) |
| **InitialCash** | 10,000.0 | Starting cash balance for portfolio |

### Portfolio Mechanics (`Portfolio.cs`)
- **Buy (action 0)**: Spends all available cash to buy shares at current price (minus fee)
- **Sell (action 1)**: Sells all holdings at current price (minus fee), converts to cash
- **Hold (action 2)**: No transaction
- **Fee**: Deducted from trade value (buy: cash outlay increases; sell: proceeds decrease)

### Reward Calculation (`RewardCalculator.cs`)
```
reward = (newPortfolioValue - oldPortfolioValue - transactionCost) × rewardScale
         - RiskAversion × maxDrawdown × currentVolatility
```
- `rewardScale = 1.0 / InitialCash` (converts dollar amounts to percentage returns)
- `maxDrawdown = (MaxPortfolioValue - CurrentPortfolioValue) / MaxPortfolioValue`
- `currentVolatility` = normalized Volatility-20 from features (index 2)

---

## 8. Summary Statistics

| Metric | Value |
|--------|-------|
| Raw CSV lines | 1,257 |
| Usable data rows | 1,255 |
| Features per timestep | 6 |
| Train steps | 1,004 |
| Test steps | 251 |
| Normalization range | [0, 1] |
| Split method | Chronological (80/20) |
| Transaction fee | 0.1% |
| Risk aversion | 1.0 |
| Initial capital | $10,000 |

---

## 9. Data Pipeline Overview

```
CSV (1257 lines)
    │
    ▼
CsvStockLoader.Load()  ──skips header + ticker row──► 1255 StockDataPoint records
    │
    ▼
FeatureEngineer.ComputeFeatures()  ──6 features × 1255 timesteps──► double[1255][6]
    │
    ▼
MinMaxNormalizer.Fit(train[0..1003]) → Transform(all)
    │
    ▼
TrainTestSplitter.Split(0.8)
    │
    ├──► trainFeatures: double[1004][6]  +  trainPrices: double[1004]
    │
    └──► testFeatures:  double[251][6]   +  testPrices:  double[251]
    │
    ▼
StockEnvironment(trainFeatures, trainPrices, fee=0.001, risk=1.0, cash=10000)
StockEnvironment(testFeatures,  testPrices,  fee=0.001, risk=1.0, cash=10000)
```