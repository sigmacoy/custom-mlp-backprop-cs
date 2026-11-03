## The src folder contains the core architecture and modular logic of the project, split into 6 modules:

- NN: Custom 3-layer Multi-Layer Perceptron (forward pass, backpropagation, MSE loss, layers).

- DQN: Reinforcement learning engine (Deep Q-Network agent, replay buffer, ϵ-greedy policy).
  
- Env: Stock trading simulation environment (portfolio tracking, buy/sell/hold execution, reward calculation).

- Data: Data pipeline (CSV loader, technical indicators/feature engineering, normalization, train-test splitting).

- Training: Training execution loop, CSV logging, and evaluation metrics calculator (Sharpe, Drawdown, Returns).

- Models: Data structure classes (e.g., historical stock quote representations).