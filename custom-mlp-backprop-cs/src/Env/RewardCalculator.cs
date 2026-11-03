namespace custom_mlp_backprop_cs.Env;


/// Calculates rewards for the DQN trading agent based on portfolio value changes, transaction costs, and risk penalties.
public static class RewardCalculator
{
    
    /// Computes the reward for a state transition.
    
    /// <param name="oldPortfolioValue">Portfolio value before the action.</param>
    /// <param name="newPortfolioValue">Portfolio value after the action.</param>
    /// <param name="transactionCost">Transaction cost incurred (fee paid). Positive value = cost/fee paid.</param>
    /// <param name="currentVolatility">Current market volatility (normalized or raw).</param>
    /// <param name="maxPortfolioValue">Peak portfolio value observed so far (for drawdown calculation).</param>
    /// <param name="riskAversion">Coefficient for risk penalty (default: 1.0). Higher values penalize risk more.</param>
    /// <param name="rewardScale">Scaling factor for the reward (e.g., 1/InitialCash for percentage returns). Default: 1.0.</param>
    /// <returns>
    /// Reward = portfolio_value_change - transaction_cost - risk_penalty
    /// where:
    /// - portfolio_value_change = newPortfolioValue - oldPortfolioValue
    /// - transaction_cost = fee paid (positive value)
    /// - risk_penalty = volatility * drawdown * risk_aversion
    /// - drawdown = (maxPortfolioValue - newPortfolioValue) / maxPortfolioValue
    /// All monetary values are scaled by rewardScale.
    /// </returns>
    /// <exception cref="ArgumentException">Thrown if maxPortfolioValue is not positive.</exception>
    public static double CalculateReward(
        double oldPortfolioValue,
        double newPortfolioValue,
        double transactionCost,
        double currentVolatility,
        double maxPortfolioValue,
        double riskAversion = 1.0,
        double rewardScale = 1.0)
    {
        if (maxPortfolioValue <= 0)
            throw new ArgumentException($"Max portfolio value must be positive, got {maxPortfolioValue}", nameof(maxPortfolioValue));
        if (rewardScale <= 0)
            throw new ArgumentException($"Reward scale must be positive, got {rewardScale}", nameof(rewardScale));

        // Portfolio value change (profit/loss) - scaled
        double portfolioValueChange = (newPortfolioValue - oldPortfolioValue) * rewardScale;

        // Transaction cost is the fee paid (always positive or zero) - scaled
        double cost = Math.Max(0, transactionCost) * rewardScale;

        // Calculate drawdown from peak
        double drawdown = (maxPortfolioValue - newPortfolioValue) / maxPortfolioValue;

        // Risk penalty: volatility * drawdown * risk_aversion
        // This encourages stable returns and penalizes large drawdowns during volatile periods
        double riskPenalty = currentVolatility * drawdown * riskAversion;

        // Total reward
        double reward = portfolioValueChange - cost - riskPenalty;

        return reward;
    }

    
    /// Computes the reward using the default risk aversion of 1.0 and no scaling.
    
    /// <param name="oldPortfolioValue">Portfolio value before the action.</param>
    /// <param name="newPortfolioValue">Portfolio value after the action.</param>
    /// <param name="transactionCost">Transaction cost incurred (fee paid).</param>
    /// <param name="currentVolatility">Current market volatility.</param>
    /// <param name="maxPortfolioValue">Peak portfolio value observed so far.</param>
    /// <returns>Calculated reward.</returns>
    public static double CalculateReward(
        double oldPortfolioValue,
        double newPortfolioValue,
        double transactionCost,
        double currentVolatility,
        double maxPortfolioValue)
    {
        return CalculateReward(oldPortfolioValue, newPortfolioValue, transactionCost, currentVolatility, maxPortfolioValue, 1.0, 1.0);
    }

    
    /// Computes the reward with custom risk aversion but default scaling.
    
    /// <param name="oldPortfolioValue">Portfolio value before the action.</param>
    /// <param name="newPortfolioValue">Portfolio value after the action.</param>
    /// <param name="transactionCost">Transaction cost incurred (fee paid).</param>
    /// <param name="currentVolatility">Current market volatility.</param>
    /// <param name="maxPortfolioValue">Peak portfolio value observed so far.</param>
    /// <param name="riskAversion">Risk aversion coefficient.</param>
    /// <returns>Calculated reward.</returns>
    public static double CalculateReward(
        double oldPortfolioValue,
        double newPortfolioValue,
        double transactionCost,
        double currentVolatility,
        double maxPortfolioValue,
        double riskAversion)
    {
        return CalculateReward(oldPortfolioValue, newPortfolioValue, transactionCost, currentVolatility, maxPortfolioValue, riskAversion, 1.0);
    }
}