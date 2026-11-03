# DQN Stock Trading Agent - Custom MLP with Backpropagation in C# (.NET 10)

## Overview

Reinforcement Learning project using a custom 3-layer MLP (Input→Hidden→Output) with backpropagation from scratch (no external ML libraries) to trade AAPL stock via Deep Q-Network (DQN).

The agent learns to trade AAPL using 5 years of historical daily data (1,255 data points) with features including SMA-5, SMA-20, 20-day volatility, and normalized price. The neural network is built entirely from scratch using a custom `Matrix` class for all linear algebra operations.

## Prerequisites

- **.NET 10 SDK** (preview or release)

## Build & Run Commands

```bash
dotnet build          # Build the project
dotnet run            # Run training (1000 episodes ~12 min on CPU)
```

## Project Structure

```
src/
├── NN/               # Neural network (Matrix, Layer, MLP, Activation)
├── DQN/              # RL components (Agent, ReplayBuffer, Experience)
├── Env/              # Trading environment (Portfolio, State, Reward, Env)
├── Data/             # CSV loading, features, normalization, splitting
├── Training/         # Training loop, metrics (Sharpe, MaxDrawdown)
├── Models/           # Data records (StockDataPoint, Experience)
└── Program.cs        # Entry point, hyperparameters, orchestration
```

### Key Components

| Folder | Purpose |
|--------|---------|
| `NN/` | Core neural network math: `Matrix` (dot, transpose, element-wise), `Layer` (weights, bias, forward/backward), `Activation` (ReLU, Linear + derivatives), `MLP` (3-layer: 6→64→3) |
| `DQN/` | Deep Q-Network: `DQNAgent` (epsilon-greedy, target network, training step), `ReplayBuffer` (ring buffer, batch sampling), `Experience` (S,A,R,S',done record) |
| `Env/` | Trading simulation: `StockEnvironment` (step/reset), `Portfolio` (cash/holdings, fees), `TradingState` (6-dim normalized vector), `RewardCalculator` (Δportfolio - fee - risk) |
| `Data/` | Data pipeline: `CsvStockLoader` (skips ticker row), `FeatureEngineer` (SMA-5, SMA-20, Volatility-20), `MinMaxNormalizer` (fit on train only), `TrainTestSplitter` (chronological 80/20) |
| `Training/` | Orchestration: `TrainingLoop` (episode loop, logging), `MetricsCalculator` (Sharpe, MaxDrawdown, Cumulative Return) |

## Key Features

- **Custom Matrix-based math** — All NN operations use `Matrix` class (no raw arrays in NN layer)
- **MinMax normalization** — Fit on TRAIN only, apply to both train and test (no data leakage)
- **Chronological train/test split** — 80/20 split preserves temporal order, prevents lookahead bias
- **Experience replay + target network** — Ring buffer (10,000 capacity), target network updated every 100 steps
- **Epsilon-greedy exploration** — Starts at 1.0, decays to 0.01 (decay factor 0.995/episode)
- **No external ML libraries** — Pure C# implementation, .NET 10 features (records, primary constructors, pattern matching)

## Results Reference

- **Training logs**: [OUTPUT-Console.MD](OUTPUT-Console.MD) — Full console output from 1000-episode training run
- **Screenshots**: [Screenshot-Console-Output/](Screenshot-Console-Output/) — Visual captures of training progress and final metrics

### Baseline Results (1000 episodes)

| Metric | Train | Test |
|--------|-------|------|
| Cumulative Return | 3.91% | 0.25% |
| Sharpe Ratio | -0.66 | -4.84 |
| Max Drawdown | 0.23% | 0.10% |
| Best Episode Return | **22.42%** | — |

*TD Loss converged to ~0.0038. Overfitting observed (train/test gap). See [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) for experiments.*

## Documentation Links

- [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) — Sensitivity experiments (learning rate, hidden size, episodes)
- [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md) — Data source, features, preprocessing details

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `HiddenSize` | 64 | Hidden layer neurons |
| `LearningRate` | 0.001 | Adam-style step size (custom SGD) |
| `Gamma` | 0.99 | Discount factor |
| `EpsilonStart` | 1.0 | Initial exploration rate |
| `EpsilonEnd` | 0.01 | Final exploration rate |
| `EpsilonDecay` | 0.995 | Per-episode decay multiplier |
| `BufferCapacity` | 10,000 | Replay buffer size |
| `BatchSize` | 32 | Minibatch size |
| `TargetUpdateFreq` | 100 | Target network sync interval (steps) |
| `Episodes` | 1000 | Training episodes |
| `TrainSplit` | 0.8 | Chronological train/test ratio |
| `TransactionFee` | 0.001 | Per-trade fee (0.1%) |
| `InitialCash` | 10,000 | Starting portfolio cash |
| `RiskAversion` | 1.0 | Risk penalty coefficient |
| `StateSize` | 6 | Input features (SMA5, SMA20, Vol20, Close, CashNorm, HoldingsNorm) |
| `ActionSize` | 3 | Actions: Buy, Sell, Hold |

---

**Built with .NET 10 • No external ML dependencies • Pure C# neural network from scratch**