# CLAUDE.md - Project Instructions

## Project: DQN Stock Trading Agent (.NET 10.0)

### Build/Run Commands
```bash
dotnet build          # Build project
dotnet run            # Run training
dotnet test           # Run tests (if added)
```

### Architecture Rules
- **No external ML libraries** - all NN math from scratch (Matrix, Layer, MLP)
- **Modular structure** - each class in own file under logical folders
- **net10.0** - use modern C# features (records, pattern matching, primary ctors)

### Folder Structure
```
src/
├── Models/           # Data records (StockDataPoint, Experience)
├── Data/             # CSV loading, features, normalization, splitting
├── NN/               # Neural network (Matrix, Layer, MLP, Activation)
├── DQN/              # RL components (Agent, ReplayBuffer)
├── Env/              # Trading environment (Portfolio, State, Reward, Env)
├── Training/         # Training loop, metrics
└── Program.cs        # Entry point
```

### Key Conventions
1. **Matrix-based math** - all NN operations use `Matrix` class (no raw arrays in NN)
2. **Records for data** - `StockDataPoint`, `Experience`, `TradingState` are `readonly record`
3. **Dependency injection style** - pass dependencies via constructor, not static access
4. **Random seed** - use single `Random` instance per class, seeded for reproducibility
5. **Logging** - `Console.WriteLine` with formatted strings, every 100 epochs
6. **Normalization** - fit MinMax on TRAIN only, apply to both train/test

### Important Files
- `docs/SPEC.md` - Full specification
- `docs/PLAN.md` - Task breakdown with dependencies
- `filemap.md` - Navigation index (update when adding/moving files)

### CSV Quirks
- Row 1: Header `Date,Close,High,Low,Open,Volume`
- Row 2: `,AAPL,AAPL,AAPL,AAPL,AAPL` - **SKIP THIS ROW**
- Data starts row 3

### Hyperparameters (in Program.cs)
```csharp
HiddenSize = 64
LearningRate = 0.001
Gamma = 0.99
EpsilonStart = 1.0
EpsilonEnd = 0.01
EpsilonDecay = 0.995
BufferCapacity = 10000
BatchSize = 32
TargetUpdateFreq = 100
Episodes = 1000
TrainSplit = 0.8
TransactionFee = 0.001
```