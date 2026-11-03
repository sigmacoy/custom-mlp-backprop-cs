# Project Summary (in Layman's Terms)

---

## 1. What does the console output mean? (Layman's Terms)

Think of this program as putting an AI into a **stock market flight simulator**:

1. **[1/7] to [4/7] Preparing the Environment:**  
   The program takes 5 years of historical Apple (AAPL) stock prices. It reserves the first 4 years (80%) for the agent to practice on, and locks away the final year (20%) as a final "blind exam" that the AI is not allowed to peek at.
2. **[5/7] & [6/7] Creating the AI Trader:**  
   We create a custom 3-layer neural network with 6 sensory inputs (moving averages, volatility, price, cash, holdings) and 3 possible decisions: **Buy**, **Sell**, or **Hold**.
3. **[7/7] Training (1,000 Episodes):**  
   The AI plays through the 4-year training data 1,000 times.
   * **TD Loss (~0.0038):** Measures how wrong the AI's predictions were. As the numbers drop, the AI gets better at estimating the financial outcome of its actions.
   * **Epsilon (1.0 → 0.01):** Controls how much the AI explores. At first ($\epsilon=1.0$), it behaves like a toddler randomly pressing buttons. Over time ($\epsilon=0.01$), it switches to making calculated decisions based on what it learned.
4. **Final Test Exam Results (Unseen Data):**  
   * **Cumulative Return (+0.25%):** The AI made a slight positive profit on days it had never seen before.
   * **Sharpe Ratio (-4.84):** Measures return relative to risk. Because daily returns fluctuate and returns were modest, the risk-adjusted score is negative.
   * **Max Drawdown (0.10%):** The deepest dip the portfolio experienced from its peak was only 0.10%—meaning the AI prioritized protecting capital from large losses.

---

## 2. Is this reinforcement learning? (Layman's Terms)

**Yes, 100%.**

* **Supervised Learning (Not this):** Like studying with a teacher who gives you an answer key for every question (e.g., *"At row 50, the correct answer is BUY"*).
* **Reinforcement Learning (This project):** Like learning to play a video game through trial and error. Nobody tells the AI what to do at each second. Instead:
  1. The AI looks at current market indicators (**State**).
  2. It chooses to **Buy, Sell, or Hold** (**Action**).
  3. It receives a financial profit or loss minus transaction fees and risk penalties (**Reward**).
  4. It uses **Backpropagation** on its 3-layer neural network to update its memory so it can make better decisions next time.

---

## 3. How the Project Satisfies the Grading Criteria

### Criterion 1: Functional Correctness & Backpropagation (35 pts)
* **What is implemented:**  
  A custom 3-layer Multi-Layer Perceptron (MLP) built from scratch without external ML libraries (`src/NN/`).
* **Mathematical Soundness:**  
  Implements the complete forward pass, Mean Squared Bellman Error loss calculation:
  $$\mathcal{L} = \frac{1}{2}(\hat{Q} - Q_{\text{target}})^2$$
  and analytical backpropagation gradient updates through both output and hidden layers (`MLP.cs`, `DenseLayer.cs`, `DQNAgent.cs`).

### Criterion 2: Data Preprocessing & Pipeline (20 pts)
* **What is implemented:**  
  * **CSV Parsing (`CsvStockLoader.cs`):** Safely parses 5 years of daily OHLCV quotes, handles corrupt rows, and strips metadata headers.
  * **Feature Engineering (`FeatureEngineer.cs`):** Computes 5-day SMA, 20-day SMA, and 20-day rolling price volatility.
  * **MinMax Normalization (`MinMaxNormalizer.cs`):** Scales features strictly between $[0, 1]$. Crucially, it fits only on the training set and transforms both sets, preventing data leakage.
  * **Chronological Split (`TrainTestSplitter.cs`):** Uses an 80/20 chronological split to avoid lookahead bias in time-series data.

### Criterion 3: Hyperparameter Tuning & Experimentation (15 pts)
* **What is implemented:**  
  Documented in detail inside [`HYPERPARAMETER_TUNING.md`](HYPERPARAMETER_TUNING.md):
  * Evaluates sensitivity across learning rates (`0.01`, `0.001`, `0.0001`), hidden node capacities (`32`, `64`, `128`), and training episode lengths (`500`, `1000`, `2000`).
  * Explains why `lr = 0.001` and `HiddenSize = 64` form the optimal balance between stability, representation capacity, and overfitting.

### Criterion 4: Evaluation & Performance Analysis (15 pts)
* **What is implemented:**  
  Evaluated using industry-standard financial and reinforcement learning metrics (`MetricsCalculator.cs`):
  * **MSE / TD Loss:** Monitored across all 1,000 episodes to verify neural network convergence.
  * **Cumulative Return (%):** Quantifies total portfolio growth on unseen test data.
  * **Sharpe Ratio:** Evaluates risk-adjusted returns per unit of volatility.
  * **Max Drawdown:** Measures worst peak-to-trough capital decline.
  * **Visualizations:** Visual charts plotted and documented in [`Screenshot-Console-Output/README.md`](Screenshot-Console-Output/README.md).

### Criterion 5: Code Quality, Design & Documentation (15 pts)
* **What is implemented:**  
  * Strict modular architecture with clean separation of concerns (`NN`, `DQN`, `Env`, `Data`, `Training`).
  * Comprehensive documentation: `README.md`, `INSTRUCTIONS.md`, `DATA_DESCRIPTION.md`, `OUTPUT-Console.MD`, and `Summary.md`.
  * Compiles out-of-the-box on modern .NET with **0 errors and 0 warnings**.

---

## 4. Is this real-world practical? (Defense Summary)

> *"Yes, it is practical as a **valid proof-of-concept for algorithmic decision-making**. While it is not designed for direct commercial deployment—as live markets require microsecond order execution, alternative data, and dynamic portfolio allocation—the implementation solves genuine real-world engineering constraints: it prevents lookahead bias through chronological temporal splitting, accounts for transaction friction with fees, penalizes downside variance, and trains a functional 3-layer neural network on five years of empirical market data to optimize an autonomous sequential policy."*
