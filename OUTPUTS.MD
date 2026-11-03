# Outputs

## With Episodes = 50
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

Starting training for 50 episodes...
State size: 6, Action size: 3, Hidden size: 64
Learning rate: 0.001, Gamma: 0.99
Epsilon: 1 -> 0.01 (decay: 0.995)
Buffer capacity: 10000, Batch size: 32
Target update frequency: 100
================================================================================
================================================================================
Training complete. Best return: 13.34%, Final return: -0.49%

Evaluating on test set...

================================================================================
FINAL EVALUATION METRICS (TEST SET)
================================================================================
Cumulative Return : 0.85%
Sharpe Ratio      : -0.8713
Max Drawdown      : 0.32%
================================================================================

TRAINING SET METRICS (for reference)
Cumulative Return : 0.98%
Sharpe Ratio      : -3.2201
Max Drawdown      : 0.10%
================================================================================
```

## Full Training with Episodes = 1000
```

```