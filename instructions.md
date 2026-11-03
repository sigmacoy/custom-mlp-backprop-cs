# Instructions: DQN Stock Trading Agent

A custom 3-layer Multi-Layer Perceptron (MLP) trained with Deep Q-Learning (DQN) and backpropagation in C# to trade Apple Inc. (AAPL) stock.

---

## Prerequisites

* **.NET SDK** (version 8.0, 9.0, or 10.0+)  
  Verify installation by running:
  ```bash
  dotnet --version
  ```

---

## How to Run

From the root directory of the repository:

### 1. Quick Test Run (~1–2 minutes)
By default, running 1,000 episodes takes ~25–30 minutes. To quickly test that everything builds and executes:

1. Open `custom-mlp-backprop-cs/Program.cs`.
2. Update line 17:
   ```csharp
   const int Episodes = 100; // Quick test mode
   ```
3. Run the project:
   ```bash
   dotnet run --project custom-mlp-backprop-cs
   ```
   *(Alternatively: `cd custom-mlp-backprop-cs && dotnet run`)*
---

### 2. Full Training Run (~25–30 minutes)
To train the full policy:

1. In `custom-mlp-backprop-cs/Program.cs`, ensure line 17 is set to:
   ```csharp
   const int Episodes = 1000;
   ```
2. Run the project:
   ```bash
   dotnet run --project custom-mlp-backprop-cs
   ```
---

## Expected Outputs

1. **Console Output:**
   * **Data Pipeline Steps:** Loading CSV, computing indicators (SMA-5, SMA-20, Volatility), MinMax normalization, and 80/20 train-test split.
   * **Training Progress:** Prints every 100 episodes showing:
     * Episode count
     * Average Reward
     * Final Portfolio Value
     * MSE Training Loss
     * Epsilon (ε) exploration rate
   * **Final Evaluation Metrics:**
     * Cumulative Return (%)
     * Sharpe Ratio
     * Maximum Drawdown (Max DD)

2. **Generated File:**
   * `training_log.csv`: Exported directly to the working directory containing per-episode records of loss, rewards, and portfolio values.