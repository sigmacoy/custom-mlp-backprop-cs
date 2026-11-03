namespace custom_mlp_backprop_cs.DQN;


/// Ring buffer implementation for experience replay in DQN.
/// Stores transitions and supports random batch sampling for training.
public class ReplayBuffer
{
    private readonly Experience[] _buffer;
    private int _head;
    private int _count;


    /// Gets the maximum capacity of the replay buffer.
    public int Capacity { get; }

    /// Gets the current number of experiences stored in the buffer.
    public int Count => _count;

    /// Initializes a new instance of the <see cref="ReplayBuffer"/> class.

    /// <param name="capacity">The maximum number of experiences to store.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when capacity is not positive.</exception>
    public ReplayBuffer(int capacity)
    {
        if (capacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be positive.");
        }

        Capacity = capacity;
        _buffer = new Experience[capacity];
        _head = 0;
        _count = 0;
    }

    /// Adds an experience to the buffer, overwriting the oldest when full (ring buffer behavior).

    /// <param name="experience">The experience to add.</param>
    public void Add(Experience experience)
    {
        _buffer[_head] = experience;
        _head = (_head + 1) % Capacity;
        if (_count < Capacity)
        {
            _count++;
        }
    }


    /// Samples a random batch of experiences from the buffer.

    /// <param name="batchSize">The number of experiences to sample.</param>
    /// <param name="rng">Optional random number generator. If null, a new instance is created.</param>
    /// <returns>
    /// A tuple containing arrays of states, actions, rewards, next states, and done flags.
    /// Each array has length equal to the actual batch size (min of batchSize and Count).
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize is not positive.</exception>
    public (double[][] States, int[] Actions, double[] Rewards, double[][] NextStates, bool[] Dones)
        Sample(int batchSize, Random? rng = null)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        }

        rng ??= new Random();

        int actualBatchSize = Math.Min(batchSize, _count);
        if (actualBatchSize == 0)
        {
            return (
                Array.Empty<double[]>(),
                Array.Empty<int>(),
                Array.Empty<double>(),
                Array.Empty<double[]>(),
                Array.Empty<bool>()
            );
        }

        // Sample without replacement using Fisher-Yates shuffle on indices
        int[] indices = new int[_count];
        for (int i = 0; i < _count; i++)
        {
            indices[i] = i;
        }

        // Partial Fisher-Yates shuffle to get random indices
        for (int i = 0; i < actualBatchSize; i++)
        {
            int j = rng.Next(i, _count);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var states = new double[actualBatchSize][];
        var actions = new int[actualBatchSize];
        var rewards = new double[actualBatchSize];
        var nextStates = new double[actualBatchSize][];
        var dones = new bool[actualBatchSize];

        for (int i = 0; i < actualBatchSize; i++)
        {
            int bufferIndex = indices[i];
            // Convert logical index to physical buffer index
            int physicalIndex = (_head - _count + bufferIndex + Capacity) % Capacity;
            var exp = _buffer[physicalIndex];

            states[i] = exp.State;
            actions[i] = exp.Action;
            rewards[i] = exp.Reward;
            nextStates[i] = exp.NextState;
            dones[i] = exp.Done;
        }

        return (states, actions, rewards, nextStates, dones);
    }


    /// Clears all experiences from the buffer.
    public void Clear()
    {
        _head = 0;
        _count = 0;
        Array.Clear(_buffer, 0, _buffer.Length);
    }
}