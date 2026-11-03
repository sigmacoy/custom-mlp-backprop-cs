namespace custom_mlp_backprop_cs.DQN;

using custom_mlp_backprop_cs.NN;

/// Deep Q-Network (DQN) Agent for stock trading.
/// Implements epsilon-greedy policy, experience replay, and target network for stable training.
public class DQNAgent
{
    /// Size of the state vector (number of features).
    public int StateSize { get; }

    /// Number of possible actions (3: Buy, Sell, Hold).
    public int ActionSize { get; }

    /// Size of the hidden layer in the neural network.
    public int HiddenSize { get; }

    /// Learning rate for weight updates.
    public double LearningRate { get; }

    /// Discount factor for future rewards (gamma).
    public double Gamma { get; }

    /// Current epsilon value for epsilon-greedy exploration.
    public double Epsilon { get; private set; }

    /// Minimum epsilon value (exploration floor).
    public double EpsilonMin { get; }

    /// Epsilon decay rate per training step.
    public double EpsilonDecay { get; }

    /// Frequency (in training steps) to update the target network.
    public int TargetUpdateFrequency { get; }

    /// Number of training steps performed.
    public int TrainStepCount { get; private set; }

    /// Batch size for training.
    public int BatchSize { get; }

    /// The online (main) network that is trained every step.
    public MLP OnlineNetwork { get; }

    /// The target network used for stable TD target computation.
    /// Updated periodically from the online network.
    public MLP TargetNetwork { get; }

    /// Experience replay buffer.
    public ReplayBuffer ReplayBuffer { get; }

    private readonly Random _rng;

    /// Initializes a new instance of the <see cref="DQNAgent"/> class.
    /// <param name="stateSize">Size of the state vector (number of features, typically 6).</param>
    /// <param name="actionSize">Number of possible actions (default: 3 for Buy/Sell/Hold).</param>
    /// <param name="hiddenSize">Size of the hidden layer (default: 64).</param>
    /// <param name="learningRate">Learning rate for gradient descent (default: 0.001).</param>
    /// <param name="gamma">Discount factor for future rewards (default: 0.99).</param>
    /// <param name="epsilonStart">Initial epsilon for exploration (default: 1.0).</param>
    /// <param name="epsilonEnd">Minimum epsilon (default: 0.01).</param>
    /// <param name="epsilonDecay">Epsilon decay factor per step (default: 0.995).</param>
    /// <param name="bufferCapacity">Replay buffer capacity (default: 10000).</param>
    /// <param name="batchSize">Training batch size (default: 32).</param>
    /// <param name="targetUpdateFrequency">Target network update frequency (default: 100).</param>
    /// <param name="rng">Optional random number generator. If null, a new instance is created.</param>
    /// <exception cref="ArgumentException">Thrown if any parameter is invalid.</exception>
    public DQNAgent(
        int stateSize,
        int actionSize = 3,
        int hiddenSize = 64,
        double learningRate = 0.001,
        double gamma = 0.99,
        double epsilonStart = 1.0,
        double epsilonEnd = 0.01,
        double epsilonDecay = 0.995,
        int bufferCapacity = 10000,
        int batchSize = 32,
        int targetUpdateFrequency = 100,
        Random? rng = null)
    {
        if (stateSize <= 0)
            throw new ArgumentException($"StateSize must be positive, got {stateSize}", nameof(stateSize));
        if (actionSize <= 0)
            throw new ArgumentException($"ActionSize must be positive, got {actionSize}", nameof(actionSize));
        if (hiddenSize <= 0)
            throw new ArgumentException($"HiddenSize must be positive, got {hiddenSize}", nameof(hiddenSize));
        if (learningRate <= 0)
            throw new ArgumentException($"LearningRate must be positive, got {learningRate}", nameof(learningRate));
        if (gamma < 0 || gamma > 1)
            throw new ArgumentException($"Gamma must be in [0, 1], got {gamma}", nameof(gamma));
        if (epsilonStart < 0 || epsilonStart > 1)
            throw new ArgumentException($"EpsilonStart must be in [0, 1], got {epsilonStart}", nameof(epsilonStart));
        if (epsilonEnd < 0 || epsilonEnd > 1)
            throw new ArgumentException($"EpsilonEnd must be in [0, 1], got {epsilonEnd}", nameof(epsilonEnd));
        if (epsilonDecay <= 0 || epsilonDecay > 1)
            throw new ArgumentException($"EpsilonDecay must be in (0, 1], got {epsilonDecay}", nameof(epsilonDecay));
        if (bufferCapacity <= 0)
            throw new ArgumentException($"BufferCapacity must be positive, got {bufferCapacity}", nameof(bufferCapacity));
        if (batchSize <= 0)
            throw new ArgumentException($"BatchSize must be positive, got {batchSize}", nameof(batchSize));
        if (targetUpdateFrequency <= 0)
            throw new ArgumentException($"TargetUpdateFrequency must be positive, got {targetUpdateFrequency}", nameof(targetUpdateFrequency));

        StateSize = stateSize;
        ActionSize = actionSize;
        HiddenSize = hiddenSize;
        LearningRate = learningRate;
        Gamma = gamma;
        Epsilon = epsilonStart;
        EpsilonMin = epsilonEnd;
        EpsilonDecay = epsilonDecay;
        TargetUpdateFrequency = targetUpdateFrequency;
        BatchSize = batchSize;
        TrainStepCount = 0;
        _rng = rng ?? new Random();

        // Create online network
        OnlineNetwork = new MLP(stateSize, hiddenSize, actionSize, _rng);

        // Create target network as a clone of online network
        TargetNetwork = OnlineNetwork.Clone(_rng);

        // Create replay buffer
        ReplayBuffer = new ReplayBuffer(bufferCapacity);
    }


    /// Selects an action using epsilon-greedy policy.

    /// <param name="state">Current state as a double array.</param>
    /// <returns>Action index: 0=Buy, 1=Sell, 2=Hold.</returns>
    /// <exception cref="ArgumentNullException">Thrown if state is null.</exception>
    /// <exception cref="ArgumentException">Thrown if state length doesn't match StateSize.</exception>
    public int SelectAction(double[] state)
    {
        if (state == null)
            throw new ArgumentNullException(nameof(state));
        if (state.Length != StateSize)
            throw new ArgumentException($"State length ({state.Length}) must match StateSize ({StateSize})", nameof(state));

        // Epsilon-greedy: explore with probability epsilon, exploit otherwise
        if (_rng.NextDouble() < Epsilon)
        {
            return _rng.Next(ActionSize);
        }

        // Exploit: choose action with highest Q-value
        double[] qValues = OnlineNetwork.GetQValues(state);
        int bestAction = 0;
        double maxQ = qValues[0];
        for (int i = 1; i < qValues.Length; i++)
        {
            if (qValues[i] > maxQ)
            {
                maxQ = qValues[i];
                bestAction = i;
            }
        }
        return bestAction;
    }


    /// Selects an action using epsilon-greedy policy (Matrix version).

    /// <param name="state">Current state as a column vector matrix.</param>
    /// <returns>Action index: 0=Buy, 1=Sell, 2=Hold.</returns>
    public int SelectAction(Matrix state)
    {
        if (state == null)
            throw new ArgumentNullException(nameof(state));
        if (state.Rows != StateSize || state.Cols != 1)
            throw new ArgumentException($"State matrix must have shape [{StateSize}, 1], got [{state.Rows}, {state.Cols}]", nameof(state));

        double[] stateArray = new double[StateSize];
        for (int i = 0; i < StateSize; i++)
        {
            stateArray[i] = state.Data[i];
        }
        return SelectAction(stateArray);
    }

    /// Performs a training step on a batch sampled from the replay buffer.
    /// Implements Double DQN: online network selects actions, target network evaluates.

    /// <param name="batchSize">Batch size (default: uses agent's BatchSize).</param>
    /// <returns>Average TD loss for the batch, or 0 if buffer has insufficient samples.</returns>
    public double TrainStep(int? batchSize = null)
    {
        int bs = batchSize ?? BatchSize;
        if (ReplayBuffer.Count < bs)
            return 0.0;

        var (states, actions, rewards, nextStates, dones) = ReplayBuffer.Sample(bs, _rng);
        double totalLoss = 0.0;

        for (int i = 0; i < states.Length; i++)
        {
            // Compute TD target
            double targetQ;
            if (dones[i])
            {
                // Terminal state: target is just the reward
                targetQ = rewards[i];
            }
            else
            {
                // Double DQN: online network selects best action, target network evaluates it
                double[] nextQOnline = OnlineNetwork.GetQValues(nextStates[i]);
                int bestNextAction = 0;
                double maxNextQOnline = nextQOnline[0];
                for (int j = 1; j < nextQOnline.Length; j++)
                {
                    if (nextQOnline[j] > maxNextQOnline)
                    {
                        maxNextQOnline = nextQOnline[j];
                        bestNextAction = j;
                    }
                }

                double[] nextQTarget = TargetNetwork.GetQValues(nextStates[i]);
                targetQ = rewards[i] + Gamma * nextQTarget[bestNextAction];
            }

            // Compute loss and gradients for the action taken
            int actionTaken = actions[i];
            double loss = OnlineNetwork.Backward(targetQ, actionTaken);
            totalLoss += loss;
        }

        // Update online network weights
        OnlineNetwork.UpdateWeights(LearningRate);

        // Increment training step counter
        TrainStepCount++;

        // Update target network periodically
        if (TrainStepCount % TargetUpdateFrequency == 0)
        {
            UpdateTargetNetwork();
        }

        // Decay epsilon
        DecayEpsilon();

        return totalLoss / states.Length;
    }


    /// Updates the target network by copying weights from the online network.
    public void UpdateTargetNetwork()
    {
        TargetNetwork.CopyWeightsFrom(OnlineNetwork);
    }

    /// Decays epsilon towards the minimum value.
    public void DecayEpsilon()
    {
        Epsilon = Math.Max(EpsilonMin, Epsilon * EpsilonDecay);
    }


    /// Sets epsilon to a specific value (e.g., for evaluation with pure exploitation).

    /// <param name="epsilon">New epsilon value, clamped to [EpsilonMin, 1].</param>
    public void SetEpsilon(double epsilon)
    {
        Epsilon = Math.Max(EpsilonMin, Math.Min(1.0, epsilon));
    }


    /// Adds an experience to the replay buffer.

    /// <param name="experience">The experience to store.</param>
    public void Remember(Experience experience)
    {
        ReplayBuffer.Add(experience);
    }

    /// Gets Q-values for a state (for evaluation/debugging).
    
    /// <param name="state">Input state as a double array.</param>
    /// <returns>Array of Q-values for each action.</returns>
    public double[] GetQValues(double[] state) => OnlineNetwork.GetQValues(state);

    /// Gets Q-values for a state (Matrix version).

    /// <param name="state">Input state as a column vector matrix.</param>
    /// <returns>Matrix of Q-values.</returns>
    public Matrix GetQValues(Matrix state) => OnlineNetwork.GetQValues(state);
}