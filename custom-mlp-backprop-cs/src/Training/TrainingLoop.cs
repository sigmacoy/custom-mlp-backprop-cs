namespace custom_mlp_backprop_cs.Training;

using custom_mlp_backprop_cs.DQN;
using custom_mlp_backprop_cs.Env;
using System.IO;

/// Training history record containing all metrics from a training run.
/// <param name="EpisodeReturns">Cumulative return for each episode.</param>
/// <param name="EpisodeLosses">Average TD loss for each episode.</param>
/// <param name="EpsilonHistory">Epsilon value at each episode.</param>
/// <param name="FinalReturn">Cumulative return of the final episode.</param>
/// <param name="FinalLoss">Average TD loss of the final episode.</param>
/// <param name="BestReturn">Best cumulative return achieved during training.</param>
public readonly record struct TrainingHistory(
    List<double> EpisodeReturns,
    List<double> EpisodeLosses,
    List<double> EpsilonHistory,
    double FinalReturn,
    double FinalLoss,
    double BestReturn
);

/// Orchestrates the DQN training loop for stock trading.
/// Manages episodes, logging, and evaluation.
public class TrainingLoop
{

    /// Total number of training episodes.
    public int Episodes { get; }


    /// Interval (in episodes) at which to log training progress.
    public int LogInterval { get; }


    /// The DQN agent being trained.
    public DQNAgent Agent { get; }


    /// The training environment.
    public StockEnvironment Environment { get; }


    /// Cumulative return for each episode.
    public List<double> EpisodeReturns { get; } = new();


    /// Average TD loss for each episode.
    public List<double> EpisodeLosses { get; } = new();


    /// Epsilon value at each episode.
    public List<double> EpsilonHistory { get; } = new();


    /// Initializes a new training loop.
    /// <param name="agent">The DQN agent to train.</param>
    /// <param name="environment">The training environment.</param>
    /// <param name="episodes">Total number of episodes (default: 1000).</param>
    /// <param name="logInterval">Logging interval in episodes (default: 100).</param>
    /// <exception cref="ArgumentNullException">Thrown if agent or environment is null.</exception>
    /// <exception cref="ArgumentException">Thrown if episodes or logInterval is not positive.</exception>
    public TrainingLoop(
        DQNAgent agent,
        StockEnvironment environment,
        int episodes = 1000,
        int logInterval = 100)
    {
        Agent = agent ?? throw new ArgumentNullException(nameof(agent));
        Environment = environment ?? throw new ArgumentNullException(nameof(environment));

        if (episodes <= 0)
            throw new ArgumentException($"Episodes must be positive, got {episodes}", nameof(episodes));
        if (logInterval <= 0)
            throw new ArgumentException($"LogInterval must be positive, got {logInterval}", nameof(logInterval));

        Episodes = episodes;
        LogInterval = logInterval;
    }


    /// Runs the complete training loop for all episodes.
    /// <returns>Training history containing all metrics.</returns>
    public TrainingHistory Run()
    {
        double bestReturn = double.MinValue;

        Console.WriteLine($"Starting training for {Episodes} episodes...");
        Console.WriteLine($"State size: {Agent.StateSize}, Action size: {Agent.ActionSize}, Hidden size: {Agent.HiddenSize}");
        Console.WriteLine($"Learning rate: {Agent.LearningRate}, Gamma: {Agent.Gamma}");
        Console.WriteLine($"Epsilon: {Agent.Epsilon} -> {Agent.EpsilonMin} (decay: {Agent.EpsilonDecay})");
        Console.WriteLine($"Buffer capacity: {Agent.ReplayBuffer.Capacity}, Batch size: {Agent.BatchSize}");
        Console.WriteLine($"Target update frequency: {Agent.TargetUpdateFrequency}");
        Console.WriteLine(new string('=', 80));

        for (int episode = 0; episode < Episodes; episode++)
        {
            var (epReturn, epLoss) = RunEpisode();

            EpisodeReturns.Add(epReturn);
            EpisodeLosses.Add(epLoss);
            EpsilonHistory.Add(Agent.Epsilon);

            if (epReturn > bestReturn)
                bestReturn = epReturn;

            // Log every LogInterval episodes
            if ((episode + 1) % LogInterval == 0)
            {
                Console.WriteLine($"Episode {episode + 1,5}/{Episodes} | " +
                    $"TD Loss: {epLoss:F6} | " +
                    $"Cumulative Return: {epReturn:P2} | " +
                    $"Epsilon: {Agent.Epsilon:F4} | " +
                    $"Best: {bestReturn:P2}");
            }
        }

        Console.WriteLine(new string('=', 80));
        Console.WriteLine($"Training complete. Best return: {bestReturn:P2}, Final return: {EpisodeReturns[^1]:P2}");

        return new TrainingHistory(
            EpisodeReturns, EpisodeLosses, EpsilonHistory,
            EpisodeReturns[^1], EpisodeLosses[^1], bestReturn);
    }


    /// Exports the training history to a CSV file for external analysis/plotting.
    /// <param name="history">The training history to export.</param>
    /// <param name="filePath">Path to the output CSV file (default: training_log.csv in current directory).</param>
    /// <exception cref="ArgumentNullException">Thrown if history is null.</exception>
    public static void ExportToCsv(TrainingHistory history, string filePath = "training_log.csv")
    {
        if (history.EpisodeReturns == null || history.EpisodeLosses == null || history.EpsilonHistory == null)
            throw new ArgumentNullException(nameof(history), "TrainingHistory lists cannot be null");

        int count = history.EpisodeReturns.Count;
        using var writer = new StreamWriter(filePath);
        writer.WriteLine("Episode,TD_Loss,Cumulative_Return,Epsilon");

        for (int i = 0; i < count; i++)
        {
            writer.WriteLine($"{i + 1},{history.EpisodeLosses[i]:F6},{history.EpisodeReturns[i]:F6},{history.EpsilonHistory[i]:F6}");
        }
    }


    /// Runs a single training episode.
    /// <returns>Tuple of (cumulative return, average TD loss).</returns>
    public (double Return, double Loss) RunEpisode()
    {
        var state = Environment.Reset();
        double totalLoss = 0;
        int steps = 0;
        bool isDone = false;

        while (!isDone)
        {
            // Select action using epsilon-greedy policy
            int action = Agent.SelectAction(state.ToArray());

            // Execute step in environment
            var (nextState, reward, stepDone) = Environment.Step(action);

            // Store experience in replay buffer
            Agent.Remember(new Experience(
                state.ToArray(),
                action,
                reward,
                nextState.ToArray(),
                stepDone
            ));

            // Train on a batch from replay buffer
            double loss = Agent.TrainStep();
            totalLoss += loss;
            steps++;

            // Update state
            state = nextState;
            isDone = stepDone;
        }

        double avgLoss = steps > 0 ? totalLoss / steps : 0;

        // Calculate cumulative return
        double finalValue = Environment.GetPortfolioValue();
        double cumReturn = (finalValue - Environment.InitialCash) / Environment.InitialCash;

        return (cumReturn, avgLoss);
    }


    /// Evaluates the agent on a test environment without exploration or training.

    /// <param name="testEnvironment">The test environment to evaluate on.</param>
    /// <returns>Tuple of (cumulative return, portfolio values at each step).</returns>
    /// <exception cref="ArgumentNullException">Thrown if testEnvironment is null.</exception>
    public (double Return, double[] PortfolioValues) Evaluate(StockEnvironment testEnvironment)
    {
        if (testEnvironment == null)
            throw new ArgumentNullException(nameof(testEnvironment));

        var state = testEnvironment.Reset();
        bool isDone = false;
        var portfolioValues = new List<double> { testEnvironment.GetPortfolioValue() };

        // Temporarily disable exploration for pure exploitation
        double oldEpsilon = Agent.Epsilon;
        Agent.SetEpsilon(0.0);

        try
        {
            while (!isDone)
            {
                int action = Agent.SelectAction(state.ToArray());
                var (nextState, reward, stepDone) = testEnvironment.Step(action);
                portfolioValues.Add(testEnvironment.GetPortfolioValue());
                state = nextState;
                isDone = stepDone;
            }
        }
        finally
        {
            // Restore epsilon
            Agent.SetEpsilon(oldEpsilon);
        }

        double finalValue = testEnvironment.GetPortfolioValue();
        double cumReturn = (finalValue - testEnvironment.InitialCash) / testEnvironment.InitialCash;

        return (cumReturn, portfolioValues.ToArray());
    }
}