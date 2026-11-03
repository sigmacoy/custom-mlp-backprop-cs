# File Map - DQN Stock Trading Agent

## Root
- `custom-mlp-backprop-cs.sln` — Solution file
- `custom-mlp-backprop-cs/` — Project root
  - `custom-mlp-backprop-cs.csproj` — Project config (net10.0)
  - `Program.cs` — Entry point, wires all components
  - `filemap.md` — This file
  - `docs/SPEC.md` — Full specification
  - `docs/PLAN.md` — Task breakdown
  - `dataset/apple_5yr_one.csv` — Raw stock data (1257 rows)
  - `.gitignore` — Git ignore rules

## Source Structure
```
src/
├── Models/
│   ├── StockDataPoint.cs     # Single OHLCV row
│   └── Experience.cs         # (state, action, reward, nextState, done)
├── Data/
│   ├── CsvStockLoader.cs     # Parse CSV, skip AAPL ticker row
│   ├── FeatureEngineer.cs    # SMA-5, SMA-20, Volatility-20
│   ├── MinMaxNormalizer.cs   # Fit on train, transform both
│   └── TrainTestSplitter.cs  # Chronological 80/20 split
├── NN/
│   ├── Matrix.cs             # 2D math: dot, transpose, element-wise
│   ├── Activation.cs         # ReLU, Linear + derivatives
│   ├── Layer.cs              # Weights, bias, forward, backward
│   └── MLP.cs                # 3-layer: Input→Hidden(ReLU)→Output(Linear, 3 Q-values)
├── DQN/
│   ├── ReplayBuffer.cs       # Ring buffer: Add, Sample(batch)
│   └── DQNAgent.cs           # Epsilon-greedy, TrainStep, TargetNetwork
├── Env/
│   ├── Portfolio.cs          # Cash, holdings, Buy/Sell/Hold with fee
│   ├── TradingState.cs       # 6-dim normalized state vector
│   ├── RewardCalculator.cs   # Portfolio Δ - fee - risk penalty
│   └── StockEnvironment.cs   # Step/Reset, wraps normalized features
├── Training/
│   ├── MetricsCalculator.cs  # Sharpe, MaxDrawdown, CumulativeReturn
│   └── TrainingLoop.cs       # Episode loop, logging every 100
```

## Legacy (did not use)
- `Sir-Aliac-Libraries/` — Old .NET 3.5 backprop code (BUT DID reference only)