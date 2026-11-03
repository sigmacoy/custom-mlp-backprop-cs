using custom_mlp_backprop_cs.Data;
using custom_mlp_backprop_cs.DQN;
using custom_mlp_backprop_cs.Env;
using custom_mlp_backprop_cs.NN;
using custom_mlp_backprop_cs.Training;

// Hyperparameters
const int HiddenSize = 64;
const double LearningRate = 0.001;
const double Gamma = 0.99;
const double EpsilonStart = 1.0;
const double EpsilonEnd = 0.01;
const double EpsilonDecay = 0.995;
const int BufferCapacity = 10000;
const int BatchSize = 32;
const int TargetUpdateFreq = 100;
const int Episodes = 1000; // for quick test, set to 50 or 100. For Full. set it to 1000
const double TrainSplit = 0.8;
const double TransactionFee = 0.001;
const double InitialCash = 10000.0;
const double RiskAversion = 1.0;
const string DataPath = "dataset/apple_5yr_one.csv";

Console.WriteLine("=".PadRight(80, '='));
Console.WriteLine("DQN Stock Trading Agent - AAPL (5 Years)");
Console.WriteLine("=".PadRight(80, '='));
Console.WriteLine();

// 1. Load raw stock data from CSV
Console.WriteLine("[1/7] Loading data from CSV...");
var rawData = CsvStockLoader.Load(DataPath);
Console.WriteLine($"       Loaded {rawData.Count} data points.");

// 2. Compute features (SMA-5, SMA-20, Volatility-20, Close)
Console.WriteLine("[2/7] Computing features...");
var features = FeatureEngineer.ComputeFeatures(rawData);
Console.WriteLine($"       Computed {features.Length} feature vectors with {features[0].Length} features each.");

// 3. Extract close prices for portfolio valuation
double[] closePrices = rawData.Select(d => d.Close).ToArray();

// 4. Normalize features (fit on train only, apply to both)
Console.WriteLine("[3/7] Normalizing features (MinMax)...");
int trainSize = (int)(features.Length * TrainSplit);
var normalizer = new MinMaxNormalizer();
normalizer.Fit(features[..trainSize]);  // Fit on TRAIN only
double[][] normalizedFeatures = normalizer.Transform(features);  // Apply to ALL
Console.WriteLine($"       Normalization complete. Train size: {trainSize}, Test size: {features.Length - trainSize}");

// 5. Chronological train/test split
Console.WriteLine("[4/7] Splitting data chronologically (80/20)...");
var (trainFeatures, testFeatures) = TrainTestSplitter.Split(normalizedFeatures, TrainSplit);
var (trainPrices, testPrices) = TrainTestSplitter.Split(closePrices, TrainSplit);
Console.WriteLine($"       Train: {trainFeatures.Length} steps, Test: {testFeatures.Length} steps");

// 6. Create environments
Console.WriteLine("[5/7] Creating environments...");
var trainEnv = new StockEnvironment(
    trainFeatures,
    trainPrices,
    TransactionFee,
    RiskAversion,
    InitialCash
);

var testEnv = new StockEnvironment(
    testFeatures,
    testPrices,
    TransactionFee,
    RiskAversion,
    InitialCash
);
Console.WriteLine("       Environments created.");

// 7. Create DQN agent
Console.WriteLine("[6/7] Initializing DQN agent...");
int stateSize = 6;  // SMA5, SMA20, Volatility20, Close, CashNorm, HoldingsNorm
int actionSize = 3; // Buy, Sell, Hold
var rng = new Random(42);  // Fixed seed for reproducibility

var agent = new DQNAgent(
    stateSize: stateSize,
    actionSize: actionSize,
    hiddenSize: HiddenSize,
    learningRate: LearningRate,
    gamma: Gamma,
    epsilonStart: EpsilonStart,
    epsilonEnd: EpsilonEnd,
    epsilonDecay: EpsilonDecay,
    bufferCapacity: BufferCapacity,
    batchSize: BatchSize,
    targetUpdateFrequency: TargetUpdateFreq,
    rng: rng
);
Console.WriteLine($"       Agent created: {stateSize} inputs -> {HiddenSize} hidden -> {actionSize} outputs");

// 8. Create training loop and run
Console.WriteLine("[7/7] Starting training...");
Console.WriteLine();

var trainingLoop = new TrainingLoop(
    agent,
    trainEnv,
    Episodes,
    100  // Log every 100 episodes
);

TrainingHistory history = trainingLoop.Run();

TrainingLoop.ExportToCsv(history, "training_log.csv");
Console.WriteLine("Training log exported to training_log.csv");

Console.WriteLine();

// 9. Evaluate on test set
Console.WriteLine("Evaluating on test set...");
var (testReturn, testPortfolioValues) = trainingLoop.Evaluate(testEnv);
var (testCumReturn, testSharpe, testMaxDD) = MetricsCalculator.CalculateAll(testPortfolioValues);

// Export test portfolio values for visualization
File.WriteAllLines("test_portfolio.csv",
    new[] { "Step,Portfolio_Value" }
        .Concat(testPortfolioValues.Select((v, i) => $"{i},{v:F2}")));
Console.WriteLine("Test portfolio values exported to test_portfolio.csv");

Console.WriteLine();
Console.WriteLine("=".PadRight(80, '='));
Console.WriteLine("FINAL EVALUATION METRICS (TEST SET)");
Console.WriteLine("=".PadRight(80, '='));
Console.WriteLine($"Cumulative Return : {testCumReturn:P2}");
Console.WriteLine($"Sharpe Ratio      : {testSharpe:F4}");
Console.WriteLine($"Max Drawdown      : {testMaxDD:P2}");
Console.WriteLine("=".PadRight(80, '='));

// Also print training set metrics for comparison
var (trainReturn, trainPortfolioValues) = trainingLoop.Evaluate(trainEnv);
var (trainCumReturn, trainSharpe, trainMaxDD) = MetricsCalculator.CalculateAll(trainPortfolioValues);

Console.WriteLine();
Console.WriteLine("TRAINING SET METRICS (for reference)");
Console.WriteLine($"Cumulative Return : {trainCumReturn:P2}");
Console.WriteLine($"Sharpe Ratio      : {trainSharpe:F4}");
Console.WriteLine($"Max Drawdown      : {trainMaxDD:P2}");
Console.WriteLine("=".PadRight(80, '='));