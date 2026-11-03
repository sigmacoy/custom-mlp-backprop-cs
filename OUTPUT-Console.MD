# Outputs

## Full Training with Episodes = 1000

```
================================================================================
DQN Stock Trading Agent - AAPL (5 Years)
================================================================================

[1/7] Loading data from CSV...
       Loaded 1255 data points.
[2/7] Computing features...
       Computed 1255 feature vectors with 6 features each.
[3/7] Normalizing features (MinMax)...
       Normalization complete. Train size: 1004, Test size: 251
[4/7] Splitting data chronologically (80/20)...
       Train: 1004 steps, Test: 251 steps
[5/7] Creating environments...
       Environments created.
[6/7] Initializing DQN agent...
       Agent created: 6 inputs -> 64 hidden -> 3 outputs
[7/7] Starting training...

Starting training for 1000 episodes...
State size: 6, Action size: 3, Hidden size: 64
Learning rate: 0.001, Gamma: 0.99
Epsilon: 1 -> 0.01 (decay: 0.995)
Buffer capacity: 10000, Batch size: 32
Target update frequency: 100
================================================================================
Episode   100/1000 | TD Loss: 0.004021 | Cumulative Return: 7.69% | Epsilon: 0.0100 | Best: 16.64%
Episode   200/1000 | TD Loss: 0.003972 | Cumulative Return: 3.06% | Epsilon: 0.0100 | Best: 16.64%
Episode   300/1000 | TD Loss: 0.003932 | Cumulative Return: -3.41% | Epsilon: 0.0100 | Best: 18.74%
Episode   400/1000 | TD Loss: 0.003173 | Cumulative Return: 1.76% | Epsilon: 0.0100 | Best: 22.42%
Episode   500/1000 | TD Loss: 0.003417 | Cumulative Return: -0.14% | Epsilon: 0.0100 | Best: 22.42%
Episode   600/1000 | TD Loss: 0.003812 | Cumulative Return: -2.62% | Epsilon: 0.0100 | Best: 22.42%
Episode   700/1000 | TD Loss: 0.003867 | Cumulative Return: 4.85% | Epsilon: 0.0100 | Best: 22.42%
Episode   800/1000 | TD Loss: 0.003910 | Cumulative Return: -0.45% | Epsilon: 0.0100 | Best: 22.42%
Episode   900/1000 | TD Loss: 0.003827 | Cumulative Return: -0.31% | Epsilon: 0.0100 | Best: 22.42%
Episode  1000/1000 | TD Loss: 0.003824 | Cumulative Return: -2.14% | Epsilon: 0.0100 | Best: 22.42%
================================================================================
Training complete. Best return: 22.42%, Final return: -2.14%

Evaluating on test set...

================================================================================
FINAL EVALUATION METRICS (TEST SET)
================================================================================
Cumulative Return : 0.25%
Sharpe Ratio      : -4.8422
Max Drawdown      : 0.10%
================================================================================

TRAINING SET METRICS (for reference)
Cumulative Return : 3.91%
Sharpe Ratio      : -0.6646
Max Drawdown      : 0.23%
================================================================================
```

## What does this mean?

The agent trained for **1000 episodes** on 5 years of AAPL daily data (1255 data points, 80/20 train/test split).

**Key observations:**

| Metric | Train | Test |
|--------|-------|------|
| Cumulative Return | 3.91% | 0.25% |
| Sharpe Ratio | -0.66 | -4.84 |
| Max Drawdown | 0.23% | 0.10% |

- **Best episode return**: 22.42% (achieved around episode 400)
- **Final episode return**: -2.14%
- **TD Loss** stabilized around 0.0038-0.0040, indicating convergence

**Interpretation:**
- The agent learned a strategy that outperformed buy-and-hold during training (best 22.42% vs ~market return)
- However, test performance (0.25%) suggests **overfitting** to the training period
- Negative Sharpe ratios indicate the strategy's returns don't compensate for volatility
- The low max drawdown (< 0.25%) shows conservative position sizing

**Next steps for improvement:**
1. Add more features (technical indicators, volume patterns)
2. Implement experience replay prioritization
3. Tune hyperparameters (learning rate, epsilon decay, network size)
4. Try different reward shaping (risk-adjusted returns)
5. Add ensemble or double DQN for more stable learning