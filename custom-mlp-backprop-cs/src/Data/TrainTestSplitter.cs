namespace custom_mlp_backprop_cs.Data;


/// Provides chronological train/test splitting for time series data.
/// Maintains temporal order (no shuffling) to prevent data leakage.

public static class TrainTestSplitter
{
    
    /// Splits features chronologically into train and test sets.
    /// Train set = first <paramref name="trainRatio"/> portion, Test set = remaining portion.
    
    /// <param name="features">Feature array of shape (N, F) where N = timesteps, F = features.</param>
    /// <param name="trainRatio">Proportion of data to use for training (default 0.8 = 80%). Must be in (0, 1).</param>
    /// <returns>A tuple of (trainFeatures, testFeatures) where both are jagged arrays.</returns>
    /// <exception cref="ArgumentNullException">Thrown if features is null.</exception>
    /// <exception cref="ArgumentException">Thrown if features is empty, trainRatio is not in (0, 1), or rows have inconsistent lengths.</exception>
    public static (double[][] TrainFeatures, double[][] TestFeatures) Split(double[][] features, double trainRatio = 0.8)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Length == 0)
            throw new ArgumentException("Features array cannot be empty", nameof(features));
        if (trainRatio <= 0.0 || trainRatio >= 1.0)
            throw new ArgumentException("trainRatio must be in (0, 1)", nameof(trainRatio));

        int featureCount = features[0].Length;
        if (featureCount == 0)
            throw new ArgumentException("Features must have at least one column", nameof(features));

        // Validate all rows have the same length
        for (int i = 1; i < features.Length; i++)
        {
            if (features[i] == null)
                throw new ArgumentException($"Row {i} is null", nameof(features));
            if (features[i].Length != featureCount)
                throw new ArgumentException($"All rows must have the same length. Row 0 has {featureCount}, row {i} has {features[i].Length}", nameof(features));
        }

        int totalSamples = features.Length;
        int trainCount = (int)Math.Floor(totalSamples * trainRatio);

        // Ensure at least 1 sample in each split
        if (trainCount == 0)
            trainCount = 1;
        if (trainCount >= totalSamples)
            trainCount = totalSamples - 1;

        var trainFeatures = new double[trainCount][];
        var testFeatures = new double[totalSamples - trainCount][];

        // Copy train portion (first trainCount rows - chronological order preserved)
        Array.Copy(features, 0, trainFeatures, 0, trainCount);

        // Copy test portion (remaining rows)
        Array.Copy(features, trainCount, testFeatures, 0, totalSamples - trainCount);

        return (trainFeatures, testFeatures);
    }


    /// Splits features and corresponding labels chronologically into train and test sets.
    /// Useful when you have both features and labels that need to be split in the same way.

    /// <param name="features">Feature array of shape (N, F).</param>
    /// <param name="labels">Label array of shape (N,) or (N, L).</param>
    /// <param name="trainRatio">Proportion of data to use for training (default 0.8 = 80%).</param>
    /// <returns>A tuple of ((trainFeatures, trainLabels), (testFeatures, testLabels)).</returns>
    /// <exception cref="ArgumentNullException">Thrown if features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown if arrays have different lengths or invalid trainRatio.</exception>
    public static ((double[][] Features, double[] Labels) Train, (double[][] Features, double[] Labels) Test) Split(
        double[][] features,
        double[] labels,
        double trainRatio = 0.8)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(labels);
        if (features.Length != labels.Length)
            throw new ArgumentException($"Features length ({features.Length}) must match labels length ({labels.Length})", nameof(labels));

        var (trainFeatures, testFeatures) = Split(features, trainRatio);

        int trainCount = trainFeatures.Length;
        var trainLabels = new double[trainCount];
        var testLabels = new double[labels.Length - trainCount];

        Array.Copy(labels, 0, trainLabels, 0, trainCount);
        Array.Copy(labels, trainCount, testLabels, 0, labels.Length - trainCount);

        return ((trainFeatures, trainLabels), (testFeatures, testLabels));
    }


    /// Splits features and corresponding 2D labels (e.g., multi-output) chronologically.

    /// <param name="features">Feature array of shape (N, F).</param>
    /// <param name="labels">Label array of shape (N, L).</param>
    /// <param name="trainRatio">Proportion of data to use for training (default 0.8 = 80%).</param>
    /// <returns>A tuple of ((trainFeatures, trainLabels), (testFeatures, testLabels)).</returns>
    /// <exception cref="ArgumentNullException">Thrown if features or labels is null.</exception>
    /// <exception cref="ArgumentException">Thrown if arrays have different lengths or invalid trainRatio.</exception>
    public static ((double[][] Features, double[][] Labels) Train, (double[][] Features, double[][] Labels) Test) Split(
        double[][] features,
        double[][] labels,
        double trainRatio = 0.8)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(labels);
        if (features.Length != labels.Length)
            throw new ArgumentException($"Features length ({features.Length}) must match labels length ({labels.Length})", nameof(labels));

        var (trainFeatures, testFeatures) = Split(features, trainRatio);

        int trainCount = trainFeatures.Length;
        var trainLabels = new double[trainCount][];
        var testLabels = new double[labels.Length - trainCount][];

        Array.Copy(labels, 0, trainLabels, 0, trainCount);
        Array.Copy(labels, trainCount, testLabels, 0, labels.Length - trainCount);

        return ((trainFeatures, trainLabels), (testFeatures, testLabels));
    }

    /// Splits a 1D array chronologically into train and test sets.

    /// <param name="array">Array to split.</param>
    /// <param name="trainRatio">Proportion of data to use for training (default 0.8 = 80%).</param>
    /// <returns>A tuple of (trainArray, testArray).</returns>
    /// <exception cref="ArgumentNullException">Thrown if array is null.</exception>
    /// <exception cref="ArgumentException">Thrown if array is empty or trainRatio is not in (0, 1).</exception>
    public static (double[] Train, double[] Test) Split(double[] array, double trainRatio = 0.8)
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Length == 0)
            throw new ArgumentException("Array cannot be empty", nameof(array));
        if (trainRatio <= 0.0 || trainRatio >= 1.0)
            throw new ArgumentException("trainRatio must be in (0, 1)", nameof(trainRatio));

        int totalSamples = array.Length;
        int trainCount = (int)Math.Floor(totalSamples * trainRatio);

        if (trainCount == 0)
            trainCount = 1;
        if (trainCount >= totalSamples)
            trainCount = totalSamples - 1;

        var trainArray = new double[trainCount];
        var testArray = new double[totalSamples - trainCount];

        Array.Copy(array, 0, trainArray, 0, trainCount);
        Array.Copy(array, trainCount, testArray, 0, totalSamples - trainCount);

        return (trainArray, testArray);
    }
}