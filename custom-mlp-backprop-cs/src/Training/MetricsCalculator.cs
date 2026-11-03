namespace custom_mlp_backprop_cs.Training;

/// Provides financial metrics calculations for portfolio evaluation.
/// Includes cumulative return, Sharpe ratio, and maximum drawdown calculations.
public static class MetricsCalculator
{

    /// Calculates cumulative return from a series of portfolio values.
    /// <param name="portfolioValues">Time series of portfolio values.</param>
    /// <returns>Cumulative return as decimal (e.g., 0.15 = 15% gain).</returns>
    /// <exception cref="ArgumentException">Thrown if portfolioValues is null or empty.</exception>
    public static double CumulativeReturn(double[] portfolioValues)
    {
        if (portfolioValues == null || portfolioValues.Length == 0)
            throw new ArgumentException("Portfolio values cannot be null or empty.", nameof(portfolioValues));

        if (portfolioValues.Length == 1)
            return 0.0;

        double initialValue = portfolioValues[0];
        double finalValue = portfolioValues[portfolioValues.Length - 1];

        if (initialValue == 0)
            return 0.0;

        return (finalValue - initialValue) / initialValue;
    }


    /// Calculates Sharpe Ratio from a series of returns.
    /// Sharpe = (mean(returns) - risk_free_rate) / std(returns) * sqrt(252)
    /// Assumes daily returns, annualizes with sqrt(252).

    /// <param name="returns">Daily returns (portfolioValue[t] / portfolioValue[t-1] - 1).</param>
    /// <param name="riskFreeRate">Annual risk-free rate (default 0.02 = 2%).</param>
    /// <returns>Annualized Sharpe Ratio.</returns>
    /// <exception cref="ArgumentException">Thrown if returns is null or empty.</exception>
    public static double SharpeRatio(double[] returns, double riskFreeRate = 0.02)
    {
        if (returns == null || returns.Length == 0)
            throw new ArgumentException("Returns cannot be null or empty.", nameof(returns));

        if (returns.Length == 1)
            return 0.0;

        double dailyRiskFree = riskFreeRate / 252.0;

        // Calculate mean return
        double sum = 0.0;
        for (int i = 0; i < returns.Length; i++)
        {
            sum += returns[i];
        }
        double meanReturn = sum / returns.Length;

        // Calculate standard deviation
        double sumSquaredDiff = 0.0;
        for (int i = 0; i < returns.Length; i++)
        {
            double diff = returns[i] - meanReturn;
            sumSquaredDiff += diff * diff;
        }
        double stdReturn = Math.Sqrt(sumSquaredDiff / returns.Length);

        // Handle zero standard deviation (all returns identical)
        if (stdReturn == 0.0)
            return 0.0;

        double sharpe = (meanReturn - dailyRiskFree) / stdReturn * Math.Sqrt(252.0);
        return sharpe;
    }


    /// Calculates Maximum Drawdown from a series of portfolio values.

    /// <param name="portfolioValues">Time series of portfolio values.</param>
    /// <returns>Maximum drawdown as decimal (e.g., 0.20 = 20% max drawdown).</returns>
    /// <exception cref="ArgumentException">Thrown if portfolioValues is null or empty.</exception>
    public static double MaxDrawdown(double[] portfolioValues)
    {
        if (portfolioValues == null || portfolioValues.Length == 0)
            throw new ArgumentException("Portfolio values cannot be null or empty.", nameof(portfolioValues));

        if (portfolioValues.Length == 1)
            return 0.0;

        double peak = portfolioValues[0];
        double maxDD = 0.0;

        for (int i = 0; i < portfolioValues.Length; i++)
        {
            double value = portfolioValues[i];
            if (value > peak)
            {
                peak = value;
            }

            if (peak > 0)
            {
                double drawdown = (peak - value) / peak;
                if (drawdown > maxDD)
                {
                    maxDD = drawdown;
                }
            }
        }

        return maxDD;
    }


    /// Calculates all metrics at once from portfolio values.
    /// <param name="portfolioValues">Time series of portfolio values.</param>
    /// <param name="riskFreeRate">Annual risk-free rate for Sharpe.</param>
    /// <returns>Tuple of (cumulativeReturn, sharpeRatio, maxDrawdown).</returns>
    /// <exception cref="ArgumentException">Thrown if portfolioValues is null or empty.</exception>
    public static (double CumulativeReturn, double SharpeRatio, double MaxDrawdown) CalculateAll(double[] portfolioValues, double riskFreeRate = 0.02)
    {
        if (portfolioValues == null || portfolioValues.Length == 0)
            throw new ArgumentException("Portfolio values cannot be null or empty.", nameof(portfolioValues));

        double cumulativeReturn = CumulativeReturn(portfolioValues);
        double maxDrawdown = MaxDrawdown(portfolioValues);

        // Calculate daily returns from portfolio values for Sharpe ratio
        double[] returns = new double[portfolioValues.Length - 1];
        for (int i = 1; i < portfolioValues.Length; i++)
        {
            if (portfolioValues[i - 1] != 0)
            {
                returns[i - 1] = portfolioValues[i] / portfolioValues[i - 1] - 1.0;
            }
            else
            {
                returns[i - 1] = 0.0;
            }
        }

        double sharpeRatio = SharpeRatio(returns, riskFreeRate);

        return (cumulativeReturn, sharpeRatio, maxDrawdown);
    }
}