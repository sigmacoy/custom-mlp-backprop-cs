namespace custom_mlp_backprop_cs.NN;

/// <summary>
/// Specifies the activation function type for a neural network layer.
/// </summary>
public enum ActivationType
{
    /// <summary>
    /// Rectified Linear Unit (ReLU) activation: max(0, x). Used for hidden layers.
    /// </summary>
    ReLU,

    /// <summary>
    /// Linear (identity) activation: f(x) = x. Used for output layers.
    /// </summary>
    Linear
}

/// <summary>
/// Represents a fully connected layer in a neural network.
/// Supports forward and backward propagation with ReLU or Linear activation.
/// </summary>
public class Layer
{
    private readonly Random _rng;
    private Matrix? _weightGradients;
    private Matrix? _biasGradients;

    /// <summary>
    /// Gets the number of input neurons (fan-in).
    /// </summary>
    public int InputSize { get; }

    /// <summary>
    /// Gets the number of output neurons (fan-out).
    /// </summary>
    public int OutputSize { get; }

    /// <summary>
    /// Gets the activation function type used by this layer.
    /// </summary>
    public ActivationType ActivationType { get; }

    /// <summary>
    /// Gets the weight matrix of shape [OutputSize, InputSize].
    /// </summary>
    public Matrix Weights { get; }

    /// <summary>
    /// Gets the bias matrix of shape [OutputSize, 1].
    /// </summary>
    public Matrix Biases { get; }

    /// <summary>
    /// Gets the last input received during forward pass, stored for backpropagation.
    /// Shape: [InputSize, 1] (column vector).
    /// </summary>
    public Matrix LastInput { get; private set; }

    /// <summary>
    /// Gets the last output produced during forward pass (after activation), stored for backpropagation.
    /// Shape: [OutputSize, 1] (column vector).
    /// </summary>
    public Matrix LastOutput { get; private set; }

    /// <summary>
    /// Gets the last pre-activation values (before activation function), stored for backpropagation.
    /// Shape: [OutputSize, 1] (column vector).
    /// </summary>
    public Matrix LastPreActivation { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Layer"/> class.
    /// </summary>
    /// <param name="inputSize">Number of input neurons (must be > 0).</param>
    /// <param name="outputSize">Number of output neurons (must be > 0).</param>
    /// <param name="activation">Activation function type for this layer.</param>
    /// <param name="rng">Optional random number generator for weight initialization. If null, a new instance is created.</param>
    /// <exception cref="ArgumentException">Thrown if inputSize or outputSize is not positive.</exception>
    public Layer(int inputSize, int outputSize, ActivationType activation, Random? rng = null)
    {
        if (inputSize <= 0)
            throw new ArgumentException($"InputSize must be positive, got {inputSize}", nameof(inputSize));
        if (outputSize <= 0)
            throw new ArgumentException($"OutputSize must be positive, got {outputSize}", nameof(outputSize));

        InputSize = inputSize;
        OutputSize = outputSize;
        ActivationType = activation;
        _rng = rng ?? new Random();

        // Initialize weights based on activation type
        double weightMin, weightMax;
        if (activation == ActivationType.ReLU)
        {
            // He initialization: uniform distribution in [-sqrt(2/fan_in), sqrt(2/fan_in)]
            double bound = Math.Sqrt(2.0 / inputSize);
            weightMin = -bound;
            weightMax = bound;
        }
        else // Linear
        {
            // Xavier initialization: uniform distribution in [-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
            double bound = Math.Sqrt(6.0 / (inputSize + outputSize));
            weightMin = -bound;
            weightMax = bound;
        }

        Weights = Matrix.Random(outputSize, inputSize, weightMin, weightMax, _rng);
        Biases = Matrix.Zeros(outputSize, 1);

        // Initialize storage for forward pass
        LastInput = Matrix.Zeros(inputSize, 1);
        LastOutput = Matrix.Zeros(outputSize, 1);
        LastPreActivation = Matrix.Zeros(outputSize, 1);
    }

    /// <summary>
    /// Performs the forward pass through this layer.
    /// </summary>
    /// <param name="input">Input matrix of shape [InputSize, 1] (column vector).</param>
    /// <returns>Output matrix of shape [OutputSize, 1] (column vector) after activation.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <exception cref="ArgumentException">Thrown if input dimensions don't match InputSize.</exception>
    public Matrix Forward(Matrix input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (input.Rows != InputSize || input.Cols != 1)
            throw new ArgumentException($"Input must have shape [{InputSize}, 1], got [{input.Rows}, {input.Cols}]", nameof(input));

        // Store input for backprop
        LastInput = new Matrix(input.Rows, input.Cols, input.Data); // Deep copy

        // preActivation = Weights * input + Biases
        Matrix preActivation = Matrix.Dot(Weights, input) + Biases;
        LastPreActivation = new Matrix(preActivation.Rows, preActivation.Cols, preActivation.Data); // Deep copy

        // Apply activation function
        Matrix output = ActivationType switch
        {
            ActivationType.ReLU => Activation.ReLU(preActivation),
            ActivationType.Linear => preActivation, // Linear is identity
            _ => throw new InvalidOperationException($"Unknown activation type: {ActivationType}")
        };

        LastOutput = new Matrix(output.Rows, output.Cols, output.Data); // Deep copy
        return output;
    }

    /// <summary>
    /// Performs the backward pass through this layer, computing gradients.
    /// </summary>
    /// <param name="gradOutput">Gradient of loss with respect to this layer's output, shape [OutputSize, 1].</param>
    /// <returns>Gradient of loss with respect to this layer's input, shape [InputSize, 1], for previous layer.</returns>
    /// <exception cref="ArgumentNullException">Thrown if gradOutput is null.</exception>
    /// <exception cref="ArgumentException">Thrown if gradOutput dimensions don't match OutputSize.</exception>
    public Matrix Backward(Matrix gradOutput)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (gradOutput.Rows != OutputSize || gradOutput.Cols != 1)
            throw new ArgumentException($"gradOutput must have shape [{OutputSize}, 1], got [{gradOutput.Rows}, {gradOutput.Cols}]", nameof(gradOutput));

        // gradPreActivation = gradOutput * ActivationDerivative(LastPreActivation) (element-wise)
        Matrix activationDerivative = ActivationType switch
        {
            ActivationType.ReLU => Activation.ReLUDerivative(LastPreActivation),
            ActivationType.Linear => Activation.LinearDerivative(LastPreActivation),
            _ => throw new InvalidOperationException($"Unknown activation type: {ActivationType}")
        };

        // Element-wise multiplication: gradOutput * activationDerivative
        Matrix gradPreActivation = new Matrix(OutputSize, 1);
        for (int i = 0; i < OutputSize; i++)
        {
            gradPreActivation.Data[i] = gradOutput.Data[i] * activationDerivative.Data[i];
        }

        // gradWeights = gradPreActivation * LastInput.Transpose() (matrix multiplication)
        // gradPreActivation: [OutputSize, 1], LastInput.Transpose(): [1, InputSize] -> result: [OutputSize, InputSize]
        Matrix lastInputTransposed = LastInput.Transpose();
        _weightGradients = Matrix.Dot(gradPreActivation, lastInputTransposed);

        // gradBiases = gradPreActivation (just copy, shape [OutputSize, 1])
        _biasGradients = new Matrix(gradPreActivation.Rows, gradPreActivation.Cols, gradPreActivation.Data);

        // gradInput = Weights.Transpose() * gradPreActivation (matrix multiplication)
        // Weights.Transpose(): [InputSize, OutputSize], gradPreActivation: [OutputSize, 1] -> result: [InputSize, 1]
        Matrix weightsTransposed = Weights.Transpose();
        Matrix gradInput = Matrix.Dot(weightsTransposed, gradPreActivation);

        return gradInput;
    }

    /// <summary>
    /// Updates the layer's weights and biases using the gradients computed during the last backward pass.
    /// </summary>
    /// <param name="learningRate">Learning rate for the update (must be > 0).</param>
    /// <exception cref="ArgumentException">Thrown if learningRate is not positive.</exception>
    /// <exception cref="InvalidOperationException">Thrown if Backward has not been called yet.</exception>
    public void UpdateWeights(double learningRate)
    {
        if (learningRate <= 0)
            throw new ArgumentException($"Learning rate must be positive, got {learningRate}", nameof(learningRate));
        if (_weightGradients == null || _biasGradients == null)
            throw new InvalidOperationException("Backward must be called before UpdateWeights");

        // Weights = Weights - learningRate * weightGradients (element-wise)
        for (int i = 0; i < Weights.Data.Length; i++)
        {
            Weights.Data[i] -= learningRate * _weightGradients.Data[i];
        }

        // Biases = Biases - learningRate * biasGradients (element-wise)
        for (int i = 0; i < Biases.Data.Length; i++)
        {
            Biases.Data[i] -= learningRate * _biasGradients.Data[i];
        }
    }

    /// <summary>
    /// Gets the weight gradients computed during the last backward pass.
    /// </summary>
    /// <returns>Weight gradients matrix of shape [OutputSize, InputSize].</returns>
    /// <exception cref="InvalidOperationException">Thrown if Backward has not been called yet.</exception>
    public Matrix GetWeightGradients()
    {
        if (_weightGradients == null)
            throw new InvalidOperationException("Backward must be called before GetWeightGradients");
        return new Matrix(_weightGradients.Rows, _weightGradients.Cols, _weightGradients.Data);
    }

    /// <summary>
    /// Gets the bias gradients computed during the last backward pass.
    /// </summary>
    /// <returns>Bias gradients matrix of shape [OutputSize, 1].</returns>
    /// <exception cref="InvalidOperationException">Thrown if Backward has not been called yet.</exception>
    public Matrix GetBiasGradients()
    {
        if (_biasGradients == null)
            throw new InvalidOperationException("Backward must be called before GetBiasGradients");
        return new Matrix(_biasGradients.Rows, _biasGradients.Cols, _biasGradients.Data);
    }
}