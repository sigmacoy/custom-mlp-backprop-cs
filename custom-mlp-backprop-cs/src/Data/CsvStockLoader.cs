using System.Globalization;
using custom_mlp_backprop_cs.Models;

namespace custom_mlp_backprop_cs.Data;

/// <summary>
/// Provides functionality to load stock data from CSV files.
/// </summary>
public static class CsvStockLoader
{
    /// <summary>
    /// Loads stock data points from a CSV file.
    /// </summary>
    /// <param name="path">The path to the CSV file.</param>
    /// <returns>A chronologically ordered list of <see cref="StockDataPoint"/>.</returns>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="FormatException">Thrown when the CSV format is invalid.</exception>
    public static List<StockDataPoint> Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"CSV file not found: {path}", path);
        }

        var dataPoints = new List<StockDataPoint>();
        var lines = File.ReadAllLines(path);

        if (lines.Length < 3)
        {
            throw new FormatException("CSV file must have at least 3 lines (header, ticker row, and at least one data row).");
        }

        // Skip line 0 (header) and line 1 (ticker row: ",AAPL,AAPL,AAPL,AAPL,AAPL")
        // Data starts from line 2 (index 2)
        for (int i = 2; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            if (string.IsNullOrWhiteSpace(line))
            {
                continue; // Skip empty lines
            }

            var columns = line.Split(',');
            if (columns.Length < 6)
            {
                // Skip malformed rows
                continue;
            }

            // Parse Date (column 0)
            if (!DateTime.TryParseExact(
                    columns[0],
                    "yyyy-MM-dd",
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.None,
                    out var date))
            {
                // Skip rows with invalid date
                continue;
            }

            // Parse numeric columns using InvariantCulture
            // CSV columns: Date,Close,High,Low,Open,Volume
            // But StockDataPoint expects: Date, Open, High, Low, Close, Volume
            if (!double.TryParse(columns[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var close) ||
                !double.TryParse(columns[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var high) ||
                !double.TryParse(columns[3], NumberStyles.Float, CultureInfo.InvariantCulture, out var low) ||
                !double.TryParse(columns[4], NumberStyles.Float, CultureInfo.InvariantCulture, out var open) ||
                !double.TryParse(columns[5], NumberStyles.Float, CultureInfo.InvariantCulture, out var volume))
            {
                // Skip rows with invalid numeric data
                continue;
            }

            dataPoints.Add(new StockDataPoint(date, open, high, low, close, volume));
        }

        // Ensure chronological order (should already be sorted in CSV, but verify)
        dataPoints.Sort((a, b) => a.Date.CompareTo(b.Date));

        return dataPoints;
    }
}