namespace custom_mlp_backprop_cs.Env;


/// Stock trading environment for DQN agent.
/// Wraps normalized feature data and manages episode lifecycle.
public class StockEnvironment
{
    
    /// Normalized feature data [N, 6] where each row contains:
    /// [SMA5, SMA20, Volatility20, ClosePrice, CashPlaceholder, HoldingsPlaceholder].
    /// The last two columns are placeholders replaced with actual portfolio state.
    public double[][] Features { get; }

    
    /// Raw close prices for portfolio valuation (same length as Features).
    public double[] ClosePrices { get; }

    
    /// Transaction fee as a decimal (e.g., 0.001 = 0.1%).
    public double TransactionFee { get; }

    
    /// Risk aversion coefficient for reward calculation.
    public double RiskAversion { get; }

    
    /// Initial cash balance.
    public double InitialCash { get; }

    
    /// Current step index in the episode (0-based).
    public int CurrentStep { get; private set; }

    
    /// Portfolio manager handling cash, holdings, and trade execution.
    public Portfolio Portfolio { get; }

    
    /// Maximum portfolio value observed during the current episode (for drawdown calculation).
    public double MaxPortfolioValue { get; private set; }

    
    /// Initializes a new stock trading environment.

    /// <param name="features">Normalized feature data [N, 6] from FeatureEngineer.</param>
    /// <param name="closePrices">Raw close prices (same length as features).</param>
    /// <param name="transactionFee">Transaction fee as a decimal (default: 0.001 = 0.1%).</param>
    /// <param name="riskAversion">Risk aversion coefficient (default: 1.0).</param>
    /// <param name="initialCash">Starting cash balance (default: 10000.0).</param>
    /// <exception cref="ArgumentException">Thrown if features and closePrices have different lengths or are empty.</exception>
    /// <exception cref="ArgumentNullException">Thrown if features or closePrices is null.</exception>
    public StockEnvironment(
        double[][] features,
        double[] closePrices,
        double transactionFee = 0.001,
        double riskAversion = 1.0,
        double initialCash = 10000.0)
    {
        if (features == null)
            throw new ArgumentNullException(nameof(features));
        if (closePrices == null)
            throw new ArgumentNullException(nameof(closePrices));
        if (features.Length == 0)
            throw new ArgumentException("Features array cannot be empty", nameof(features));
        if (closePrices.Length == 0)
            throw new ArgumentException("Close prices array cannot be empty", nameof(closePrices));
        if (features.Length != closePrices.Length)
            throw new ArgumentException($"Features length ({features.Length}) must match close prices length ({closePrices.Length})", nameof(closePrices));
        if (transactionFee < 0)
            throw new ArgumentException($"Transaction fee must be non-negative, got {transactionFee}", nameof(transactionFee));
        if (riskAversion < 0)
            throw new ArgumentException($"Risk aversion must be non-negative, got {riskAversion}", nameof(riskAversion));
        if (initialCash < 0)
            throw new ArgumentException($"Initial cash must be non-negative, got {initialCash}", nameof(initialCash));

        Features = features;
        ClosePrices = closePrices;
        TransactionFee = transactionFee;
        RiskAversion = riskAversion;
        InitialCash = initialCash;

        Portfolio = new Portfolio(initialCash, transactionFee);
        MaxPortfolioValue = initialCash;
        CurrentStep = 0;
    }

    
    /// Resets the environment to the initial state.

    /// <returns>Initial TradingState with first step's features and portfolio state.</returns>
    public TradingState Reset()
    {
        Portfolio.Reset();
        CurrentStep = 0;
        MaxPortfolioValue = InitialCash;

        return CreateStateFromStep(0);
    }

    
    /// Executes one environment step with the given action.

    /// <param name="action">Action to execute: 0 = Buy, 1 = Sell, 2 = Hold.</param>
    /// <returns>
    /// A tuple containing:
    /// - NextState: TradingState for the next step (or terminal state if done).
    /// - Reward: Calculated reward for the transition.
    /// - Done: True if episode has terminated (end of data reached).
    /// </returns>
    /// <exception cref="ArgumentException">Thrown if action is invalid or environment not initialized.</exception>
    public (TradingState NextState, double Reward, bool Done) Step(int action)
    {
        if (action < 0 || action > 2)
            throw new ArgumentException($"Action must be 0 (Buy), 1 (Sell), or 2 (Hold), got {action}", nameof(action));
        if (CurrentStep >= Features.Length)
            throw new InvalidOperationException("Cannot step after episode is done. Call Reset() first.");

        // Get current price
        double currentPrice = ClosePrices[CurrentStep];

        // Get old portfolio value before action
        double oldValue = Portfolio.GetValue(currentPrice);

        // Execute action
        var (executed, cost) = Portfolio.ExecuteAction(action, currentPrice);

        // Get new portfolio value after action
        double newValue = Portfolio.GetValue(currentPrice);

        // Update max portfolio value for drawdown calculation
        if (newValue > MaxPortfolioValue)
            MaxPortfolioValue = newValue;

        // Get current volatility (normalized from features)
        // Index 2 = Volatility20
        double volatility = Features[CurrentStep][2];

        // Calculate reward
        // cost is negative for buy (outflow), positive for sell (inflow)
        // RewardCalculator expects transactionCost as positive fee paid
        double transactionCost = Math.Abs(cost);
        // Scale reward by 1/InitialCash to convert dollar amounts to percentage returns
        double rewardScale = 1.0 / InitialCash;
        double reward = RewardCalculator.CalculateReward(
            oldValue, newValue, transactionCost, volatility, MaxPortfolioValue, RiskAversion, rewardScale);

        // Advance step
        CurrentStep++;

        // Check if done (end of data)
        bool done = CurrentStep >= Features.Length - 1;

        // Create next state
        TradingState nextState;
        if (!done)
        {
            // Use next step's features + current portfolio state
            nextState = CreateStateFromStep(CurrentStep);
        }
        else
        {
            // Terminal state - use last step's features
            nextState = CreateStateFromStep(Features.Length - 1);
        }

        return (nextState, reward, done);
    }

    
    /// Gets the current total portfolio value.

    /// <returns>Total portfolio value = Cash + Holdings * currentPrice.</returns>
    public double GetPortfolioValue()
    {
        if (CurrentStep >= ClosePrices.Length)
            return Portfolio.GetValue(ClosePrices[^1]);

        return Portfolio.GetValue(ClosePrices[CurrentStep]);
    }

    
    /// Creates a TradingState from the given step index using current portfolio values.

    /// <param name="step">Step index to create state from.</param>
    /// <returns>TradingState with features from step and normalized portfolio values.</returns>
    private TradingState CreateStateFromStep(int step)
    {
        // Features[step] has 6 values: [SMA5, SMA20, Volatility20, Close, CashPlaceholder, HoldingsPlaceholder]
        // Replace last 2 with normalized portfolio values

        // Normalize cash: ratio of current cash to initial cash (range [0, 1] typically)
        double cashNorm = Portfolio.Cash / InitialCash;

        // Normalize holdings: ratio of holdings value to initial cash
        // Holdings * currentPrice / InitialCash
        double currentPrice = ClosePrices[Math.Min(step, ClosePrices.Length - 1)];
        double holdingsNorm = (Portfolio.Holdings * currentPrice) / InitialCash;

        // Clamp to reasonable bounds [0, 2] to handle edge cases
        cashNorm = Math.Max(0, Math.Min(2, cashNorm));
        holdingsNorm = Math.Max(0, Math.Min(2, holdingsNorm));

        return new TradingState(
            Features[step][0],  // SMA5
            Features[step][1],  // SMA20
            Features[step][2],  // Volatility20
            Features[step][3],  // ClosePrice
            cashNorm,
            holdingsNorm
        );
    }
}