namespace custom_mlp_backprop_cs.Data;


/// Min-Max normalizer that scales features to the [0, 1] range.
/// Fits on training data (computes min/max per feature column), then transforms both train and test data.

public sealed class MinMaxNormalizer
{
    /// Minimum value per feature column (computed during Fit).
    public double[]? MinValues { get; private set; }

    /// Maximum value per feature column (computed during Fit).
    public double[]? MaxValues { get; private set; }

    /// Number of features (columns) the normalizer was fitted on.
    public int FeatureCount => MinValues?.Length ?? 0;

    /// Indicates whether the normalizer has been fitted.
    public bool IsFitted => MinValues != null && MaxValues != null;

    /// Computes the minimum and maximum values for each feature column from the training data.
    /// NaN values are ignored during min/max computation.

    /// <param name="trainData">Training data as a jagged array of shape (N, F) where N = samples, F = features.</param>
    /// <exception cref="ArgumentNullException">Thrown if trainData is null.</exception>
    /// <exception cref="ArgumentException">Thrown if trainData is empty or rows have inconsistent lengths.</exception>
    /// <exception cref="InvalidOperationException">Thrown if a feature column contains only NaN values.</exception>
    public void Fit(double[][] trainData)
    {
        ArgumentNullException.ThrowIfNull(trainData);
        if (trainData.Length == 0)
            throw new ArgumentException("Training data cannot be empty", nameof(trainData));

        int featureCount = trainData[0].Length;
        if (featureCount == 0)
            throw new ArgumentException("Training data must have at least one feature column", nameof(trainData));

        // Validate all rows have the same length
        for (int i = 1; i < trainData.Length; i++)
        {
            if (trainData[i] == null)
                throw new ArgumentException($"Row {i} is null", nameof(trainData));
            if (trainData[i].Length != featureCount)
                throw new ArgumentException($"All rows must have the same length. Row 0 has {featureCount}, row {i} has {trainData[i].Length}", nameof(trainData));
        }

        var minValues = new double[featureCount];
        var maxValues = new double[featureCount];
        var hasValidValue = new bool[featureCount];

        // Initialize min/max
        for (int f = 0; f < featureCount; f++)
        {
            minValues[f] = double.MaxValue;
            maxValues[f] = double.MinValue;
            hasValidValue[f] = false;
        }

        // Compute min/max per feature column, ignoring NaN
        for (int i = 0; i < trainData.Length; i++)
        {
            var row = trainData[i];
            for (int f = 0; f < featureCount; f++)
            {
                double val = row[f];
                if (!double.IsNaN(val))
                {
                    if (val < minValues[f]) minValues[f] = val;
                    if (val > maxValues[f]) maxValues[f] = val;
                    hasValidValue[f] = true;
                }
            }
        }

        // Check for columns with no valid values
        for (int f = 0; f < featureCount; f++)
        {
            if (!hasValidValue[f])
            {
                throw new InvalidOperationException($"Feature column {f} contains only NaN values. Cannot compute min/max.");
            }

            // Handle case where min == max (constant feature)
            if (minValues[f] == maxValues[f])
            {
                // Expand range slightly to avoid division by zero
                // This will map all values to 0.5 after normalization
                minValues[f] -= 1e-8;
                maxValues[f] += 1e-8;
            }
        }

        MinValues = minValues;
        MaxValues = maxValues;
    }

    /// Transforms the input data using the fitted min/max values, scaling each feature to [0, 1].
    /// NaN values are preserved as NaN in the output.
    /// 
    /// <param name="data">Data to transform as a jagged array of shape (N, F).</param>

    /// <returns>Normalized data with values in [0, 1] (NaN preserved).</returns>

    /// <exception cref="InvalidOperationException">Thrown if the normalizer has not been fitted.</exception>
    /// <exception cref="ArgumentNullException">Thrown if data is null.</exception>
    /// <exception cref="ArgumentException">Thrown if data is empty or feature count doesn't match fitted normalizer.</exception>
    public double[][] Transform(double[][] data)
    {
        if (!IsFitted)
            throw new InvalidOperationException("Normalizer must be fitted before calling Transform. Call Fit() first.");

        ArgumentNullException.ThrowIfNull(data);
        if (data.Length == 0)
            return []; // Empty array is valid, return empty

        int featureCount = data[0].Length;
        if (featureCount != FeatureCount)
            throw new ArgumentException($"Data has {featureCount} features but normalizer was fitted on {FeatureCount} features", nameof(data));

        // Validate all rows have the same length
        for (int i = 1; i < data.Length; i++)
        {
            if (data[i] == null)
                throw new ArgumentException($"Row {i} is null", nameof(data));
            if (data[i].Length != featureCount)
                throw new ArgumentException($"All rows must have the same length. Row 0 has {featureCount}, row {i} has {data[i].Length}", nameof(data));
        }

        var result = new double[data.Length][];
        for (int i = 0; i < data.Length; i++)
        {
            var row = data[i];
            var normalizedRow = new double[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                double val = row[f];
                if (double.IsNaN(val))
                {
                    normalizedRow[f] = double.NaN;
                }
                else
                {
                    // Min-max normalization: (x - min) / (max - min)
                    normalizedRow[f] = (val - MinValues![f]) / (MaxValues![f] - MinValues![f]);
                    // Clamp to [0, 1] to handle numerical edge cases
                    if (normalizedRow[f] < 0.0) normalizedRow[f] = 0.0;
                    if (normalizedRow[f] > 1.0) normalizedRow[f] = 1.0;
                }
            }
            result[i] = normalizedRow;
        }

        return result;
    }

    /// Convenience method that fits the normalizer on the training data and immediately transforms it.

    /// <param name="trainData">Training data as a jagged array of shape (N, F).</param>
    /// <returns>Normalized training data with values in [0, 1].</returns>
    /// <exception cref="ArgumentNullException">Thrown if trainData is null.</exception>
    /// <exception cref="ArgumentException">Thrown if trainData is empty or rows have inconsistent lengths.</exception>
    /// <exception cref="InvalidOperationException">Thrown if a feature column contains only NaN values.</exception>
    public double[][] FitTransform(double[][] trainData)
    {
        Fit(trainData);
        return Transform(trainData);
    }

    /// Inverse transforms normalized data back to the original scale.
    /// Useful for interpreting model predictions in original units.

    /// <param name="normalizedData">Normalized data in [0, 1] range.</param>
    /// <returns>Data in original scale.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the normalizer has not been fitted.</exception>
    /// <exception cref="ArgumentNullException">Thrown if normalizedData is null.</exception>
    /// <exception cref="ArgumentException">Thrown if feature count doesn't match fitted normalizer.</exception>
    public double[][] InverseTransform(double[][] normalizedData)
    {
        if (!IsFitted)
            throw new InvalidOperationException("Normalizer must be fitted before calling InverseTransform. Call Fit() first.");

        ArgumentNullException.ThrowIfNull(normalizedData);
        if (normalizedData.Length == 0)
            return [];

        int featureCount = normalizedData[0].Length;
        if (featureCount != FeatureCount)
            throw new ArgumentException($"Data has {featureCount} features but normalizer was fitted on {FeatureCount} features", nameof(normalizedData));

        for (int i = 1; i < normalizedData.Length; i++)
        {
            if (normalizedData[i] == null)
                throw new ArgumentException($"Row {i} is null", nameof(normalizedData));
            if (normalizedData[i].Length != featureCount)
                throw new ArgumentException($"All rows must have the same length. Row 0 has {featureCount}, row {i} has {normalizedData[i].Length}", nameof(normalizedData));
        }

        var result = new double[normalizedData.Length][];
        for (int i = 0; i < normalizedData.Length; i++)
        {
            var row = normalizedData[i];
            var originalRow = new double[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                double val = row[f];
                if (double.IsNaN(val))
                {
                    originalRow[f] = double.NaN;
                }
                else
                {
                    // Inverse min-max: x * (max - min) + min
                    originalRow[f] = val * (MaxValues![f] - MinValues![f]) + MinValues![f];
                }
            }
            result[i] = originalRow;
        }

        return result;
    }
}