# DQN Stock Trading Agent - Specification

## Project Overview
Build a Deep Q-Learning (DQN) stock trading agent from scratch in C# (.NET 10.0) for a university assignment. The agent trades Apple stock (AAPL) using historical data from `apple_5yr_one.csv`.

## Goals
- Implement a custom 3-layer MLP (Input → Hidden → Output) with forward/backpropagation from scratch (no ML libraries)
- Build complete DQN components: Agent (epsilon-greedy), Experience Replay, Target Network
- Create stock trading environment with feature engineering (SMA, volatility)
- Implement reward system with risk-adjusted returns
- Train and evaluate with console logging (epoch, TD loss, cumulative return)
- Output final metrics: Cumulative Return, Sharpe Ratio, Maximum Drawdown

## Non-Goals
- No external ML libraries (ML.NET, TensorFlow, PyTorch bindings)
- No real-time trading or broker API integration
- No hyperparameter optimization framework

## Stack
- **Language**: C# 13 (.NET 10.0)
- **Build**: `dotnet build`
- **Run**: `dotnet run`
- **Test**: `dotnet test` (if tests added)
- **Target Framework**: net10.0

## Architecture - Modules & Responsibilities

### 1. Data Layer
| Class | File | Responsibility |
|-------|------|----------------|
| `StockDataPoint` | `Models/StockDataPoint.cs` | Single row: Date, Open, High, Low, Close, Volume |
| `CsvStockLoader` | `Data/CsvStockLoader.cs` | Parse CSV, handle header quirks (AAPL ticker row) |
| `FeatureEngineer` | `Data/FeatureEngineer.cs` | Compute SMA, historical volatility |
| `MinMaxNormalizer` | `Data/MinMaxNormalizer.cs` | Fit on train, transform train+test |
| `TrainTestSplitter` | `Data/TrainTestSplitter.cs` | Chronological split (e.g., 80/20) |

### 2. Neural Network Layer
| Class | File | Responsibility |
|-------|------|----------------|
| `Matrix` | `NN/Matrix.cs` | 2D array wrapper: dot, transpose, add, multiply, apply func |
| `Layer` | `NN/Layer.cs` | Weights, biases, activation (ReLU for hidden, Linear for output) |
| `MLP` | `NN/MLP.cs` | 3-layer network: Forward(), Backward(), UpdateWeights() |
| `Activation` | `NN/Activation.cs` | Static: ReLU, ReLUDerivative, Linear, LinearDerivative |

### 3. DQN Layer
| Class | File | Responsibility |
|-------|------|----------------|
| `Experience` | `DQN/Experience.cs` | Tuple: State, Action, Reward, NextState, Done |
| `ReplayBuffer` | `DQN/ReplayBuffer.cs` | Ring buffer: Add(), Sample(batchSize) |
| `DQNAgent` | `DQN/DQNAgent.cs` | Epsilon-greedy policy, TrainStep(), UpdateTargetNetwork() |
| `Transition` | `DQN/Transition.cs` | Alias for Experience (clarity) |

### 4. Environment Layer
| Class | File | Responsibility |
|-------|------|----------------|
| `TradingState` | `Env/TradingState.cs` | State vector: SMA, volatility, price, cash, holdings |
| `StockEnvironment` | `Env/StockEnvironment.cs` | Step(action) → (nextState, reward, done), Reset() |
| `RewardCalculator` | `Env/RewardCalculator.cs` | Portfolio change - transaction cost - risk penalty |
| `Portfolio` | `Env/Portfolio.cs` | Cash, holdings, value, transaction logic |

### 5. Training & Evaluation
| Class | File | Responsibility |
|-------|------|----------------|
| `TrainingLoop` | `Training/TrainingLoop.cs` | Episode loop, logging every 100 epochs |
| `MetricsCalculator` | `Training/MetricsCalculator.cs` | Sharpe, Max Drawdown, Cumulative Return |
| `Program` | `Program.cs` | Entry point, wire dependencies, run |

## Data Model
- **CSV Columns**: Date, Close, High, Low, Open, Volume (row 2 has "AAPL" ticker labels - skip)
- **Features per timestep**: SMA(5), SMA(20), Volatility(20), Close, Cash, Holdings → 6-dim state
- **Actions**: 0=Buy, 1=Sell, 2=Hold
- **Q-values**: 3 outputs from MLP

## Reward System
```
reward = portfolio_value_change 
         - transaction_cost (e.g., 0.1% per trade)
         - risk_penalty (volatility * drawdown_factor)
```

## Training Hyperparameters (Configurable)
- Hidden layer size: 64
- Learning rate: 0.001
- Gamma (discount): 0.99
- Epsilon start: 1.0, end: 0.01, decay: 0.995
- Replay buffer capacity: 10,000
- Batch size: 32
- Target network update frequency: 100 steps
- Episodes: 1000 (configurable)
- Train/Test split: 80/20 chronological

## Definition of Done (MVP)
1. Project builds with `dotnet build` (net10.0)
2. CSV loads correctly (skips header row with AAPL)
3. Features computed (SMA, volatility) and normalized
4. MLP forward/backprop works (unit test: gradient check)
5. DQN agent selects actions, stores experiences, samples batches
6. Target network updates periodically
7. Training loop runs N episodes, logs every 100 epochs
8. Final evaluation prints: Cumulative Return, Sharpe Ratio, Max Drawdown
9. Code is modular, documented, follows C# best practices