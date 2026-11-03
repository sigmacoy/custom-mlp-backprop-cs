namespace custom_mlp_backprop_cs.Models;

/// <summary>
/// Represents a single OHLCV data point for a stock.
/// </summary>
/// <param name="Date">The trading date.</param>
/// <param name="Open">Opening price.</param>
/// <param name="High">Highest price during the day.</param>
/// <param name="Low">Lowest price during the day.</param>
/// <param name="Close">Closing price.</param>
/// <param name="Volume">Trading volume.</param>
public readonly record struct StockDataPoint(
    DateTime Date,
    double Open,
    double High,
    double Low,
    double Close,
    double Volume
);