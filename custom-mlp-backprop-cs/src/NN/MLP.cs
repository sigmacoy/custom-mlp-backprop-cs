namespace custom_mlp_backprop_cs.NN;

/// <summary>
/// Multi-Layer Perceptron (MLP) with one hidden layer for DQN-based stock trading.
/// Architecture: Input (6 features) → Hidden (ReLU) → Output (3 actions: Buy/Sell/Hold, Linear).
/// </summary>
public class MLP
{
    /// <summary>
    /// Gets the number of input neurons (state features, typically 6 for stock trading).
    /// </summary>
    public int InputSize { get; }

    /// <summary>
    /// Gets the number of hidden neurons.
    /// </summary>
    public int HiddenSize { get; }

    /// <summary>
    /// Gets the number of output neurons (always 3 for Buy/Sell/Hold actions).
    /// </summary>
    public int OutputSize { get; }

    /// <summary>
    /// Gets the hidden layer (InputSize → HiddenSize, ReLU activation).
    /// </summary>
    public Layer HiddenLayer { get; }

    /// <summary>
    /// Gets the output layer (HiddenSize → OutputSize, Linear activation).
    /// </summary>
    public Layer OutputLayer { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="MLP"/> class.
    /// </summary>
    /// <param name="inputSize">Number of input features (must be > 0, typically 6 for stock state).</param>
    /// <param name="hiddenSize">Number of hidden neurons (must be > 0).</param>
    /// <param name="outputSize">Number of output actions (default 3 for Buy/Sell/Hold, must be > 0).</param>
    /// <param name="rng">Optional random number generator for weight initialization. If null, a new instance is created.</param>
    /// <exception cref="ArgumentException">Thrown if any size parameter is not positive.</exception>
    public MLP(int inputSize, int hiddenSize, int outputSize = 3, Random? rng = null)
    {
        if (inputSize <= 0)
            throw new ArgumentException($"InputSize must be positive, got {inputSize}", nameof(inputSize));
        if (hiddenSize <= 0)
            throw new ArgumentException($"HiddenSize must be positive, got {hiddenSize}", nameof(hiddenSize));
        if (outputSize <= 0)
            throw new ArgumentException($"OutputSize must be positive, got {outputSize}", nameof(outputSize));

        InputSize = inputSize;
        HiddenSize = hiddenSize;
        OutputSize = outputSize;

        // Use the same RNG for both layers to ensure consistent initialization if needed
        // But we can also create a new one from the seed
        var layerRng = rng ?? new Random();

        // Hidden layer: InputSize -> HiddenSize with ReLU activation
        HiddenLayer = new Layer(inputSize, hiddenSize, ActivationType.ReLU, layerRng);

        // Output layer: HiddenSize -> OutputSize with Linear activation
        // Use a derived RNG for the output layer to get different random weights
        var outputRng = rng != null ? new Random(rng.Next()) : new Random();
        OutputLayer = new Layer(hiddenSize, outputSize, ActivationType.Linear, outputRng);
    }

    /// <summary>
    /// Performs the forward pass through the network: state vector (6 features) → Q-values (3 actions).
    /// </summary>
    /// <param name="input">Input matrix of shape [InputSize, 1] representing the state.</param>
    /// <returns>Output matrix of shape [OutputSize, 1] containing Q-values for each action.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <exception cref="ArgumentException">Thrown if input dimensions don't match InputSize.</exception>
    public Matrix Forward(Matrix input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // Hidden layer forward pass (ReLU activation)
        Matrix hiddenOutput = HiddenLayer.Forward(input);

        // Output layer forward pass (Linear activation -> Q-values)
        Matrix output = OutputLayer.Forward(hiddenOutput);

        return output;
    }

    /// <summary>
    /// Performs the forward pass through the network using a double array.
    /// </summary>
    /// <param name="state">Input state array of length InputSize (typically 6 features).</param>
    /// <returns>Array of length OutputSize (3) containing Q-values for Buy/Sell/Hold.</returns>
    /// <exception cref="ArgumentNullException">Thrown if state is null.</exception>
    /// <exception cref="ArgumentException">Thrown if state length doesn't match InputSize.</exception>
    public double[] Forward(double[] state)
    {
        if (state == null)
            throw new ArgumentNullException(nameof(state));
        if (state.Length != InputSize)
            throw new ArgumentException($"State length ({state.Length}) must match InputSize ({InputSize})", nameof(state));

        // Convert to column vector matrix
        Matrix input = Matrix.FromColumnVectors(new[] { state });
        Matrix output = Forward(input);

        // Convert back to array
        double[] result = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            result[i] = output.Data[i];
        }
        return result;
    }

    /// <summary>
    /// Performs the backward pass for DQN training, computing TD loss and gradients for ONE action.
    /// This implements the Bellman equation gradient: computes loss for the action taken only.
    /// </summary>
    /// <param name="targetQ">Target Q-value from the Bellman equation: reward + gamma * max(Q_target(next_state)).</param>
    /// <param name="actionTaken">Index of the action that was taken (0=Buy, 1=Sell, 2=Hold).</param>
    /// <returns>The TD loss: 0.5 * (Q_pred - Q_target)^2 for the action taken.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if actionTaken is not in [0, OutputSize).</exception>
    /// <exception cref="InvalidOperationException">Thrown if Forward has not been called yet.</exception>
    public double Backward(double targetQ, int actionTaken)
    {
        if (actionTaken < 0 || actionTaken >= OutputSize)
            throw new ArgumentOutOfRangeException(nameof(actionTaken), $"actionTaken must be in [0, {OutputSize}), got {actionTaken}");

        // Get the predicted Q-value for the action taken from the last forward pass
        double qPred = OutputLayer.LastOutput[actionTaken, 0];

        // Compute TD loss: 0.5 * (Q_pred - Q_target)^2 for the action taken
        double loss = 0.5 * (qPred - targetQ) * (qPred - targetQ);

        // Create gradient output vector: zeros everywhere except at actionTaken
        // gradOutput = Q_pred - Q_target (only for actionTaken, zeros elsewhere)
        var gradOutput = Matrix.Zeros(OutputSize, 1);
        gradOutput[actionTaken, 0] = qPred - targetQ;

        // Backward through output layer
        Matrix gradHidden = OutputLayer.Backward(gradOutput);

        // Backward through hidden layer
        HiddenLayer.Backward(gradHidden);

        return loss;
    }

    /// <summary>
    /// Updates all layer weights and biases using the gradients computed during the last backward pass.
    /// </summary>
    /// <param name="learningRate">Learning rate for the update (must be > 0).</param>
    /// <exception cref="ArgumentException">Thrown if learningRate is not positive.</exception>
    /// <exception cref="InvalidOperationException">Thrown if Backward has not been called yet.</exception>
    public void UpdateWeights(double learningRate)
    {
        if (learningRate <= 0)
            throw new ArgumentException($"Learning rate must be positive, got {learningRate}", nameof(learningRate));

        HiddenLayer.UpdateWeights(learningRate);
        OutputLayer.UpdateWeights(learningRate);
    }

    /// <summary>
    /// Gets Q-values for a state (convenience method, same as Forward).
    /// </summary>
    /// <param name="state">Input state array of length InputSize.</param>
    /// <returns>Array of length OutputSize containing Q-values for each action.</returns>
    public double[] GetQValues(double[] state) => Forward(state);

    /// <summary>
    /// Gets Q-values for a state (convenience method, same as Forward).
    /// </summary>
    /// <param name="state">Input state matrix of shape [InputSize, 1].</param>
    /// <returns>Matrix of shape [OutputSize, 1] containing Q-values for each action.</returns>
    public Matrix GetQValues(Matrix state) => Forward(state);

    /// <summary>
    /// Creates a deep copy of this MLP with the same architecture and weights.
    /// Used for creating target networks in DQN.
    /// </summary>
    /// <param name="rng">Optional random number generator. If null, a new instance is created.</param>
    /// <returns>A new MLP instance with identical weights and architecture.</returns>
    public MLP Clone(Random? rng = null)
    {
        var clone = new MLP(InputSize, HiddenSize, OutputSize, rng);

        // Copy weights and biases from this MLP to the clone
        CopyWeightsTo(clone);

        return clone;
    }

    /// <summary>
    /// Copies weights and biases from another MLP (source) into this MLP.
    /// Used for updating target network from the main network in DQN.
    /// </summary>
    /// <param name="source">The source MLP to copy weights from. Must have identical architecture.</param>
    /// <exception cref="ArgumentNullException">Thrown if source is null.</exception>
    /// <exception cref="ArgumentException">Thrown if source architecture doesn't match this MLP.</exception>
    public void CopyWeightsFrom(MLP source)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (source.InputSize != InputSize)
            throw new ArgumentException($"Source InputSize ({source.InputSize}) doesn't match this MLP ({InputSize})", nameof(source));
        if (source.HiddenSize != HiddenSize)
            throw new ArgumentException($"Source HiddenSize ({source.HiddenSize}) doesn't match this MLP ({HiddenSize})", nameof(source));
        if (source.OutputSize != OutputSize)
            throw new ArgumentException($"Source OutputSize ({source.OutputSize}) doesn't match this MLP ({OutputSize})", nameof(source));

        CopyWeightsTo(source);
    }

    /// <summary>
    /// Internal helper to copy weights from this MLP to another MLP.
    /// </summary>
    /// <param name="target">The target MLP to copy weights to.</param>
    private void CopyWeightsTo(MLP target)
    {
        // Copy hidden layer weights and biases
        Array.Copy(HiddenLayer.Weights.Data, target.HiddenLayer.Weights.Data, HiddenLayer.Weights.Data.Length);
        Array.Copy(HiddenLayer.Biases.Data, target.HiddenLayer.Biases.Data, HiddenLayer.Biases.Data.Length);

        // Copy output layer weights and biases
        Array.Copy(OutputLayer.Weights.Data, target.OutputLayer.Weights.Data, OutputLayer.Weights.Data.Length);
        Array.Copy(OutputLayer.Biases.Data, target.OutputLayer.Biases.Data, OutputLayer.Biases.Data.Length);
    }
}