# DQN Stock Trading Agent - Implementation Plan

## Task List (Dependency Order)

### Task 1: Project Scaffold & Configuration (Light)
**Brief**: Update .csproj for net10.0, create folder structure, add `.gitignore`, verify build.
**Files**: `custom-mlp-backprop-cs.csproj`, folder structure under `src/`, `.gitignore`
**Verification**: `dotnet build` succeeds, folders exist
**Tier**: Light

### Task 2: Data Models & CSV Loader (Medium)
**Brief**: Implement `StockDataPoint` record and `CsvStockLoader` to parse the CSV (skip row 2 with AAPL tickers, parse Date/OHLCV).
**Files**: `Models/StockDataPoint.cs`, `Data/CsvStockLoader.cs`
**Verification**: Load CSV, print first 5 rows with correct types
**Tier**: Medium
**Depends on**: Task 1

### Task 3: Feature Engineering & Normalization (Medium)
**Brief**: Implement `FeatureEngineer` (SMA-5, SMA-20, Volatility-20), `MinMaxNormalizer` (fit on train, transform both), `TrainTestSplitter` (chronological 80/20).
**Files**: `Data/FeatureEngineer.cs`, `Data/MinMaxNormalizer.cs`, `Data/TrainTestSplitter.cs`
**Verification**: Features computed for all rows, normalized to [0,1], split indices correct
**Tier**: Medium
**Depends on**: Task 2

### Task 4: Matrix & Math Utilities (Careful)
**Brief**: Implement `Matrix` class with: constructor(rows, cols), indexer, Dot(), Transpose(), Add(), Multiply(), Apply(), ApplyInPlace(), ToArray(). This is the foundation for all NN math.
**Files**: `NN/Matrix.cs`
**Verification**: Unit tests for each operation (dot product, transpose, element-wise)
**Tier**: Careful
**Depends on**: Task 1

### Task 5: Activation Functions (Light)
**Brief**: Static `Activation` class with ReLU, ReLUDerivative, Linear, LinearDerivative.
**Files**: `NN/Activation.cs`
**Verification**: Quick manual test in Program.cs
**Tier**: Light
**Depends on**: Task 1

### Task 6: Layer Class (Careful)
**Brief**: Implement `Layer` with weights (Matrix), biases (Matrix), Forward(input), Backward(gradOutput, learningRate). Supports ReLU (hidden) and Linear (output).
**Files**: `NN/Layer.cs`
**Verification**: Forward/backward shape consistency, gradient check vs numerical
**Tier**: Careful
**Depends on**: Task 4, Task 5

### Task 7: MLP (3-Layer Network) (Careful)
**Brief**: Implement `MLP` with Input→Hidden→Output layers. Methods: `Forward(state) → Q-values[3]`, `Backward(targetQ, actionTaken) → loss`, `UpdateWeights(lr)`. Output layer: 3 neurons (Buy/Sell/Hold).
**Files**: `NN/MLP.cs`
**Verification**: Forward returns 3 values, backward computes TD loss, weights update
**Tier**: Careful
**Depends on**: Task 6

### Task 8: DQN Core - Experience & ReplayBuffer (Medium)
**Brief**: `Experience` record (state, action, reward, nextState, done). `ReplayBuffer` with capacity, `Add()`, `Sample(batchSize)` returning arrays.
**Files**: `DQN/Experience.cs`, `DQN/ReplayBuffer.cs`
**Verification**: Add 1000 experiences, sample batch returns correct shapes
**Tier**: Medium
**Depends on**: Task 1

### Task 9: DQN Agent (Heavy)
**Brief**: `DQNAgent` with: policy (epsilon-greedy decay), `SelectAction(state)`, `TrainStep(batch)`, `UpdateTargetNetwork()`. Uses two MLPs (online + target). Implements TD target: `r + γ * max_a' Q_target(s', a')`.
**Files**: `DQN/DQNAgent.cs`
**Verification**: Action selection explores/exploits correctly, TD loss decreases over time, target network updates
**Tier**: Heavy
**Depends on**: Task 7, Task 8

### Task 10: Trading Environment - Portfolio & State (Medium)
**Brief**: `Portfolio` (cash, holdings, Buy/Sell/Hold with 0.1% fee), `TradingState` (6-dim vector), `RewardCalculator` (portfolio Δ - fee - risk penalty).
**Files**: `Env/Portfolio.cs`, `Env/TradingState.cs`, `Env/RewardCalculator.cs`
**Verification**: Portfolio value tracks correctly, reward components compute
**Tier**: Medium
**Depends on**: Task 3

### Task 11: Stock Environment (Medium)
**Brief**: `StockEnvironment` wrapping normalized feature data. `Reset()` → initial state. `Step(action)` → (nextState, reward, done). Handles episode termination at data end.
**Files**: `Env/StockEnvironment.cs`
**Verification**: Step advances correctly, done=true at end, state dims match MLP input
**Tier**: Medium
**Depends on**: Task 10

### Task 12: Metrics Calculator (Medium)
**Brief**: `MetricsCalculator` with: `CumulativeReturn(portfolioValues)`, `SharpeRatio(returns)`, `MaxDrawdown(portfolioValues)`.
**Files**: `Training/MetricsCalculator.cs`
**Verification**: Known input sequences produce correct outputs
**Tier**: Medium
**Depends on**: Task 1

### Task 13: Training Loop (Heavy)
**Brief**: `TrainingLoop` orchestrating: for each episode, env.Reset(), while !done: agent.SelectAction(), env.Step(), agent.TrainStep(), log every 100 episodes. Tracks cumulative return per episode.
**Files**: `Training/TrainingLoop.cs`
**Verification**: Runs 1000 episodes, logs every 100, loss decreases, return improves
**Tier**: Heavy
**Depends on**: Task 9, Task 11, Task 12

### Task 14: Program.cs - Main Entry Point (Medium)
**Brief**: Wire all components: load data → features → normalize → split → create env/agent → train → evaluate on test → print final metrics.
**Files**: `Program.cs`
**Verification**: `dotnet run` completes, prints epoch/loss/return every 100, final metrics
**Tier**: Medium
**Depends on**: Task 13

### Task 15: Code Review & Polish (Careful)
**Brief**: Review all code for: null safety, disposal, naming consistency, XML docs on public APIs, remove legacy Sir-Aliac-Libraries reference, update filemap.md.
**Files**: All
**Verification**: `dotnet build` clean, code reads well, filemap.md current
**Tier**: Careful
**Depends on**: Task 14

---

## Parallelization Notes
- Tasks 2, 4, 5 can start in parallel after Task 1
- Tasks 6, 8 can run in parallel after their deps
- Tasks 10, 12 can run in parallel after Task 3, 1
- Task 15 runs last

## Tier Summary
- Light: 2 tasks (1, 5)
- Medium: 6 tasks (2, 3, 8, 10, 11, 12, 14)
- Careful: 4 tasks (4, 6, 7, 15)
- Heavy: 2 tasks (9, 13)