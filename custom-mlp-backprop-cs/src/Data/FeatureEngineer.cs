using custom_mlp_backprop_cs.Models;

namespace custom_mlp_backprop_cs.Data;

/// <summary>
/// Computes technical features from raw stock data points for use in ML models.
/// </summary>
public static class FeatureEngineer
{
    /// <summary>
    /// Computes a set of 6 features for each timestep from a chronologically sorted list of stock data points.
    /// </summary>
    /// <param name="dataPoints">Chronologically sorted list of stock data points (oldest first).</param>
    /// <returns>
    /// A jagged array of shape (N, 6) where N = dataPoints.Count.
    /// Features per row: [SMA-5, SMA-20, Volatility-20, Close, CashPlaceholder, HoldingsPlaceholder].
    /// NaN values from insufficient lookback data are forward-filled with the first valid value.
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown if dataPoints is null.</exception>
    /// <exception cref="ArgumentException">Thrown if dataPoints is empty.</exception>
    public static double[][] ComputeFeatures(List<StockDataPoint> dataPoints)
    {
        ArgumentNullException.ThrowIfNull(dataPoints);
        if (dataPoints.Count == 0)
            throw new ArgumentException("Data points list cannot be empty", nameof(dataPoints));

        int n = dataPoints.Count;
        var features = new double[n][];

        // Pre-extract close prices for efficiency
        var closePrices = new double[n];
        for (int i = 0; i < n; i++)
        {
            closePrices[i] = dataPoints[i].Close;
        }

        // Compute daily returns: (Close[t] - Close[t-1]) / Close[t-1]
        // Returns[0] = NaN (no previous day)
        var dailyReturns = new double[n];
        dailyReturns[0] = double.NaN;
        for (int i = 1; i < n; i++)
        {
            double prevClose = closePrices[i - 1];
            if (prevClose != 0.0)
            {
                dailyReturns[i] = (closePrices[i] - prevClose) / prevClose;
            }
            else
            {
                dailyReturns[i] = double.NaN;
            }
        }

        // Compute features for each timestep
        for (int i = 0; i < n; i++)
        {
            var row = new double[6];

            // Feature 0: SMA-5 (5-day Simple Moving Average of Close)
            row[0] = ComputeSMA(closePrices, i, 5);

            // Feature 1: SMA-20 (20-day Simple Moving Average of Close)
            row[1] = ComputeSMA(closePrices, i, 20);

            // Feature 2: Volatility-20 (20-day standard deviation of daily returns)
            row[2] = ComputeVolatility(dailyReturns, i, 20);

            // Feature 3: Close price (raw, will be normalized later)
            row[3] = closePrices[i];

            // Feature 4: Cash balance placeholder (filled by environment at runtime)
            row[4] = 0.0;

            // Feature 5: Holdings placeholder (filled by environment at runtime)
            row[5] = 0.0;

            features[i] = row;
        }

        // Forward-fill NaN values with first valid value for each feature column
        // This handles initial rows where SMA/Volatility have insufficient data
        ForwardFillNaN(features);

        return features;
    }

    /// <summary>
    /// Forward-fills NaN values in the feature array with the first valid value for each column.
    /// </summary>
    /// <param name="features">Jagged array of shape (N, F) to fill in place.</param>
    private static void ForwardFillNaN(double[][] features)
    {
        if (features.Length == 0 || features[0].Length == 0)
            return;

        int n = features.Length;
        int fCount = features[0].Length;

        for (int f = 0; f < fCount; f++)
        {
            double? firstValid = null;

            // Find first valid value
            for (int i = 0; i < n; i++)
            {
                if (!double.IsNaN(features[i][f]))
                {
                    firstValid = features[i][f];
                    break;
                }
            }

            // If no valid value found, leave as NaN (shouldn't happen with our data)
            if (!firstValid.HasValue)
                continue;

            // Fill NaN values before first valid with the first valid value
            for (int i = 0; i < n; i++)
            {
                if (double.IsNaN(features[i][f]))
                {
                    features[i][f] = firstValid.Value;
                }
                else
                {
                    // Once we hit a valid value, stop filling (forward fill only fills leading NaNs)
                    break;
                }
            }
        }
    }

    /// <summary>
    /// Computes the Simple Moving Average (SMA) for the given window ending at index <paramref name="endIndex"/> (inclusive).
    /// Uses expanding window for indices < windowSize - 1 (i.e., averages all available data up to that point).
    /// Returns NaN if no data is available.
    /// </summary>
    /// <param name="values">Array of values to compute SMA over.</param>
    /// <param name="endIndex">The ending index (inclusive) for the SMA window.</param>
    /// <param name="windowSize">The SMA window size (e.g., 5 for SMA-5, 20 for SMA-20).</param>
    /// <returns>The SMA value, or double.NaN if insufficient data.</returns>
    private static double ComputeSMA(double[] values, int endIndex, int windowSize)
    {
        if (endIndex < 0 || endIndex >= values.Length)
            return double.NaN;

        // Number of available data points up to and including endIndex
        int availableCount = endIndex + 1;
        int count = Math.Min(windowSize, availableCount);

        if (count == 0)
            return double.NaN;

        double sum = 0.0;
        int validCount = 0;

        // Sum the last 'count' values (from endIndex - count + 1 to endIndex inclusive)
        for (int i = endIndex - count + 1; i <= endIndex; i++)
        {
            double val = values[i];
            if (!double.IsNaN(val))
            {
                sum += val;
                validCount++;
            }
        }

        if (validCount == 0)
            return double.NaN;

        return sum / validCount;
    }

    /// <summary>
    /// Computes the rolling standard deviation (volatility) of daily returns for the given window.
    /// Uses expanding window for indices < windowSize - 1.
    /// Returns NaN if fewer than 2 valid return values are available in the window.
    /// </summary>
    /// <param name="dailyReturns">Array of daily returns (first element is NaN).</param>
    /// <param name="endIndex">The ending index (inclusive) for the volatility window.</param>
    /// <param name="windowSize">The volatility window size (e.g., 20 for Volatility-20).</param>
    /// <returns>The standard deviation of returns in the window, or double.NaN if insufficient data.</returns>
    private static double ComputeVolatility(double[] dailyReturns, int endIndex, int windowSize)
    {
        if (endIndex < 1 || endIndex >= dailyReturns.Length)
            return double.NaN; // Need at least one return (index 1 onwards)

        // Number of available return data points up to endIndex (returns start at index 1)
        int availableCount = endIndex; // returns[1] to returns[endIndex] inclusive = endIndex values
        int count = Math.Min(windowSize, availableCount);

        if (count < 2)
            return double.NaN; // Need at least 2 values for std dev

        // Collect valid returns in the window
        var windowReturns = new List<double>();
        for (int i = endIndex - count + 1; i <= endIndex; i++)
        {
            double ret = dailyReturns[i];
            if (!double.IsNaN(ret))
            {
                windowReturns.Add(ret);
            }
        }

        if (windowReturns.Count < 2)
            return double.NaN;

        // Compute mean
        double mean = 0.0;
        foreach (double ret in windowReturns)
        {
            mean += ret;
        }
        mean /= windowReturns.Count;

        // Compute variance (sample variance: divide by N-1)
        double variance = 0.0;
        foreach (double ret in windowReturns)
        {
            double diff = ret - mean;
            variance += diff * diff;
        }
        variance /= (windowReturns.Count - 1);

        return Math.Sqrt(variance);
    }
}