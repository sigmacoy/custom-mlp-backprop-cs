namespace custom_mlp_backprop_cs.Env;

/// Represents the normalized trading state as a 6-dimensional vector for the DQN agent.
/// All values are normalized to be suitable for neural network input.

/// <param name="SMA5">Normalized 5-period Simple Moving Average (typically in range [0, 1] or [-1, 1]).</param>
/// <param name="SMA20">Normalized 20-period Simple Moving Average (typically in range [0, 1] or [-1, 1]).</param>
/// <param name="Volatility20">Normalized 20-period volatility (typically in range [0, 1]).</param>
/// <param name="ClosePrice">Normalized closing price (typically in range [0, 1] or [-1, 1]).</param>
/// <param name="Cash">Normalized cash balance (typically in range [0, 1]).</param>
/// <param name="Holdings">Normalized holdings (number of shares, typically in range [0, 1]).</param>
public readonly record struct TradingState(
    double SMA5,
    double SMA20,
    double Volatility20,
    double ClosePrice,
    double Cash,
    double Holdings
)
{

    /// Returns the state as a flat array of 6 doubles for neural network input.
    
    /// <returns>Array of length 6 containing [SMA5, SMA20, Volatility20, ClosePrice, Cash, Holdings].</returns>
    public double[] ToArray()
    {
        return new double[6] { SMA5, SMA20, Volatility20, ClosePrice, Cash, Holdings };
    }


    /// Creates a TradingState from a flat array of 6 doubles.
    
    /// <param name="values">Array of length 6 containing [SMA5, SMA20, Volatility20, ClosePrice, Cash, Holdings].</param>
    /// <returns>A new TradingState instance.</returns>
    /// <exception cref="ArgumentException">Thrown if the array length is not 6.</exception>
    public static TradingState FromArray(double[] values)
    {
        if (values == null)
            throw new ArgumentNullException(nameof(values));
        if (values.Length != 6)
            throw new ArgumentException($"Expected array of length 6, got {values.Length}", nameof(values));

        return new TradingState(
            values[0],  // SMA5
            values[1],  // SMA20
            values[2],  // Volatility20
            values[3],  // ClosePrice
            values[4],  // Cash
            values[5]   // Holdings
        );
    }


    /// Returns a string representation of the trading state.
    
    /// <returns>Formatted string showing all 6 dimensions.</returns>
    public override string ToString()
    {
        return $"TradingState(SMA5={SMA5:G4}, SMA20={SMA20:G4}, Vol20={Volatility20:G4}, Close={ClosePrice:G4}, Cash={Cash:G4}, Holdings={Holdings:G4})";
    }
}