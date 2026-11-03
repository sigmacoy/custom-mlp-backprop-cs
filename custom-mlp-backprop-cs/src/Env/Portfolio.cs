namespace custom_mlp_backprop_cs.Env;

/// <summary>
/// Manages a trading portfolio including cash balance, stock holdings, and trade execution.
/// </summary>
public class Portfolio
{
    /// <summary>
    /// Current cash balance available for trading.
    /// </summary>
    public double Cash { get; private set; }

    /// <summary>
    /// Number of shares currently held.
    /// </summary>
    public double Holdings { get; private set; }

    /// <summary>
    /// Initial cash balance when the portfolio was created.
    /// </summary>
    public double InitialCash { get; }

    /// <summary>
    /// Transaction fee as a decimal (e.g., 0.001 = 0.1%).
    /// </summary>
    public double TransactionFee { get; }

    /// <summary>
    /// Initializes a new portfolio with the specified starting cash and transaction fee.
    /// </summary>
    /// <param name="initialCash">Starting cash balance (default: 10000.0).</param>
    /// <param name="transactionFee">Transaction fee as a decimal (default: 0.001 = 0.1%).</param>
    /// <exception cref="ArgumentException">Thrown if initialCash is negative or transactionFee is negative.</exception>
    public Portfolio(double initialCash = 10000.0, double transactionFee = 0.001)
    {
        if (initialCash < 0)
            throw new ArgumentException($"Initial cash must be non-negative, got {initialCash}", nameof(initialCash));
        if (transactionFee < 0)
            throw new ArgumentException($"Transaction fee must be non-negative, got {transactionFee}", nameof(transactionFee));

        InitialCash = initialCash;
        TransactionFee = transactionFee;
        Cash = initialCash;
        Holdings = 0.0;
    }

    /// <summary>
    /// Calculates the current total portfolio value at the given price.
    /// </summary>
    /// <param name="currentPrice">Current stock price per share.</param>
    /// <returns>Total portfolio value = Cash + Holdings * currentPrice.</returns>
    /// <exception cref="ArgumentException">Thrown if currentPrice is negative.</exception>
    public double GetValue(double currentPrice)
    {
        if (currentPrice < 0)
            throw new ArgumentException($"Current price must be non-negative, got {currentPrice}", nameof(currentPrice));

        return Cash + Holdings * currentPrice;
    }

    /// <summary>
    /// Executes a trading action (Buy, Sell, or Hold) at the current price.
    /// </summary>
    /// <param name="action">Action to execute: 0 = Buy, 1 = Sell, 2 = Hold.</param>
    /// <param name="currentPrice">Current stock price per share.</param>
    /// <returns>
    /// A tuple containing:
    /// - executed: true if the action resulted in a trade, false for Hold or invalid actions.
    /// - cost: For Buy, the total cost including fee (negative = cash outflow).
    ///         For Sell, the net revenue after fee (positive = cash inflow).
    ///         For Hold, 0.
    /// </returns>
    /// <exception cref="ArgumentException">Thrown if currentPrice is negative or action is invalid.</exception>
    public (bool executed, double cost) ExecuteAction(int action, double currentPrice)
    {
        if (currentPrice < 0)
            throw new ArgumentException($"Current price must be non-negative, got {currentPrice}", nameof(currentPrice));
        if (action < 0 || action > 2)
            throw new ArgumentException($"Action must be 0 (Buy), 1 (Sell), or 2 (Hold), got {action}", nameof(action));

        return action switch
        {
            0 => ExecuteBuy(currentPrice),    // Buy
            1 => ExecuteSell(currentPrice),   // Sell
            2 => (false, 0.0),                // Hold
            _ => (false, 0.0)                 // Should not reach here due to validation
        };
    }

    /// <summary>
    /// Resets the portfolio to its initial state (initial cash, zero holdings).
    /// </summary>
    public void Reset()
    {
        Cash = InitialCash;
        Holdings = 0.0;
    }

    /// <summary>
    /// Executes a buy order: spends all available cash (minus fee) to buy shares.
    /// </summary>
    private (bool executed, double cost) ExecuteBuy(double currentPrice)
    {
        if (Cash <= 0 || currentPrice <= 0)
            return (false, 0.0);

        // Calculate how many shares we can buy with all cash (minus fee)
        // fee = shares * price * TransactionFee
        // total cost = shares * price + fee = shares * price * (1 + TransactionFee)
        // shares = Cash / (price * (1 + TransactionFee))
        double sharesToBuy = Cash / (currentPrice * (1.0 + TransactionFee));

        if (sharesToBuy <= 0)
            return (false, 0.0);

        double tradeValue = sharesToBuy * currentPrice;
        double fee = tradeValue * TransactionFee;
        double totalCost = tradeValue + fee;

        // Execute the buy
        Cash -= totalCost;
        Holdings += sharesToBuy;

        // Return negative cost to indicate cash outflow
        return (true, -totalCost);
    }

    /// <summary>
    /// Executes a sell order: sells all holdings at current price (minus fee).
    /// </summary>
    private (bool executed, double cost) ExecuteSell(double currentPrice)
    {
        if (Holdings <= 0 || currentPrice <= 0)
            return (false, 0.0);

        double tradeValue = Holdings * currentPrice;
        double fee = tradeValue * TransactionFee;
        double netRevenue = tradeValue - fee;

        // Execute the sell
        Cash += netRevenue;
        Holdings = 0.0;

        // Return positive cost to indicate cash inflow
        return (true, netRevenue);
    }
}