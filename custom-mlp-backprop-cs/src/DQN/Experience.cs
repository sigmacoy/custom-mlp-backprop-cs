namespace custom_mlp_backprop_cs.DQN;

/// <summary>
/// Represents a single transition experience in the replay buffer.
/// </summary>
/// <param name="State">The state vector (6 features: SMA-5, SMA-20, Volatility-20, Close, Cash, Holdings).</param>
/// <param name="Action">The action taken: 0=Buy, 1=Sell, 2=Hold.</param>
/// <param name="Reward">The scalar reward received.</param>
/// <param name="NextState">The next state vector (6 features).</param>
/// <param name="Done">Whether the episode has terminated.</param>
public readonly record struct Experience(
    double[] State,
    int Action,
    double Reward,
    double[] NextState,
    bool Done
);