namespace custom_mlp_backprop_cs.NN;

/// Represents a mathematical matrix with row-major storage.
/// All binary operations return new matrices (immutable), while InPlace methods mutate the current instance.
/// </summary>
public class Matrix
{

    /// Number of rows in the matrix.
    public int Rows { get; }


    /// Number of columns in the matrix.
    public int Cols { get; }


    /// Row-major flat array storing matrix data. Index = row * Cols + col.
    public double[] Data { get; }


    /// Creates a new matrix with the specified dimensions, initialized to initValue.
    /// <param name="rows">Number of rows (must be >= 0).</param>
    /// <param name="cols">Number of columns (must be >= 0).</param>
    /// <param name="initValue">Initial value for all elements (default 0.0).</param>
    /// <exception cref="ArgumentException">Thrown if rows or cols is negative.</exception>
    public Matrix(int rows, int cols, double initValue = 0.0)
    {
        if (rows < 0)
            throw new ArgumentException($"Rows must be non-negative, got {rows}", nameof(rows));
        if (cols < 0)
            throw new ArgumentException($"Cols must be non-negative, got {cols}", nameof(cols));

        Rows = rows;
        Cols = cols;
        Data = new double[rows * cols];

        if (initValue != 0.0)
        {
            Array.Fill(Data, initValue);
        }
    }


    /// Creates a new matrix from a flat array (row-major order).
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="data">Flat array of length rows * cols in row-major order.</param>
    /// <exception cref="ArgumentException">Thrown if data length doesn't match rows * cols.</exception>
    public Matrix(int rows, int cols, double[] data)
    {
        if (rows < 0)
            throw new ArgumentException($"Rows must be non-negative, got {rows}", nameof(rows));
        if (cols < 0)
            throw new ArgumentException($"Cols must be non-negative, got {cols}", nameof(cols));
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        if (data.Length != rows * cols)
            throw new ArgumentException($"Data length ({data.Length}) must equal rows * cols ({rows * cols})", nameof(data));

        Rows = rows;
        Cols = cols;
        Data = data;
    }


    /// Creates a matrix from an array of row vectors (jagged array).
    /// <param name="rows">Jagged array where each inner array is a row.</param>
    /// <returns>A new Matrix with dimensions (rows.Length x rows[0].Length).</returns>
    /// <exception cref="ArgumentException">Thrown if rows is null, empty, or rows have inconsistent lengths.</exception>
    public static Matrix FromRowVectors(double[][] rows)
    {
        if (rows == null)
            throw new ArgumentNullException(nameof(rows));
        if (rows.Length == 0)
            return new Matrix(0, 0);

        int rowCount = rows.Length;
        int colCount = rows[0].Length;

        if (colCount == 0)
            return new Matrix(rowCount, 0);

        var data = new double[rowCount * colCount];
        for (int r = 0; r < rowCount; r++)
        {
            if (rows[r] == null)
                throw new ArgumentException($"Row {r} is null", nameof(rows));
            if (rows[r].Length != colCount)
                throw new ArgumentException($"All rows must have the same length. Row 0 has {colCount}, row {r} has {rows[r].Length}", nameof(rows));

            Array.Copy(rows[r], 0, data, r * colCount, colCount);
        }

        return new Matrix(rowCount, colCount, data);
    }


    /// Creates a matrix from an array of column vectors (jagged array).
    /// <param name="cols">Jagged array where each inner array is a column.</param>
    /// <returns>A new Matrix with dimensions (cols[0].Length x cols.Length).</returns>
    /// <exception cref="ArgumentException">Thrown if cols is null, empty, or columns have inconsistent lengths.</exception>
    public static Matrix FromColumnVectors(double[][] cols)
    {
        if (cols == null)
            throw new ArgumentNullException(nameof(cols));
        if (cols.Length == 0)
            return new Matrix(0, 0);

        int colCount = cols.Length;
        int rowCount = cols[0].Length;

        if (rowCount == 0)
            return new Matrix(0, colCount);

        var data = new double[rowCount * colCount];
        for (int c = 0; c < colCount; c++)
        {
            if (cols[c] == null)
                throw new ArgumentException($"Column {c} is null", nameof(cols));
            if (cols[c].Length != rowCount)
                throw new ArgumentException($"All columns must have the same length. Column 0 has {rowCount}, column {c} has {cols[c].Length}", nameof(cols));

            for (int r = 0; r < rowCount; r++)
            {
                data[r * colCount + c] = cols[c][r];
            }
        }

        return new Matrix(rowCount, colCount, data);
    }


    /// Gets or sets the element at the specified row and column.
    /// <param name="row">Row index (0-based).</param>
    /// <param name="col">Column index (0-based).</param>
    /// <returns>The element at [row, col].</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown if row or col is out of bounds.</exception>
    public double this[int row, int col]
    {
        get
        {
            if (row < 0 || row >= Rows || col < 0 || col >= Cols)
                throw new IndexOutOfRangeException($"Index [{row}, {col}] out of bounds for matrix {Rows}x{Cols}");
            return Data[row * Cols + col];
        }
        set
        {
            if (row < 0 || row >= Rows || col < 0 || col >= Cols)
                throw new IndexOutOfRangeException($"Index [{row}, {col}] out of bounds for matrix {Rows}x{Cols}");
            Data[row * Cols + col] = value;
        }
    }


    /// Element-wise addition of two matrices.
    /// <param name="a">Left matrix.</param>
    /// <param name="b">Right matrix.</param>
    /// <returns>A new matrix containing the element-wise sum.</returns>
    /// <exception cref="ArgumentException">Thrown if matrices have different dimensions.</exception>
    public static Matrix operator +(Matrix a, Matrix b)
    {
        ValidateSameShape(a, b, nameof(a), nameof(b));
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Data.Length; i++)
        {
            result.Data[i] = a.Data[i] + b.Data[i];
        }
        return result;
    }


    /// Element-wise subtraction of two matrices.
    /// <param name="a">Left matrix.</param>
    /// <param name="b">Right matrix.</param>
    /// <returns>A new matrix containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown if matrices have different dimensions.</exception>
    public static Matrix operator -(Matrix a, Matrix b)
    {
        ValidateSameShape(a, b, nameof(a), nameof(b));
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Data.Length; i++)
        {
            result.Data[i] = a.Data[i] - b.Data[i];
        }
        return result;
    }


    /// Element-wise multiplication (Hadamard product) of two matrices.
    /// <param name="a">Left matrix.</param>
    /// <param name="b">Right matrix.</param>
    /// <returns>A new matrix containing the element-wise product.</returns>
    /// <exception cref="ArgumentException">Thrown if matrices have different dimensions.</exception>
    public static Matrix operator *(Matrix a, Matrix b)
    {
        ValidateSameShape(a, b, nameof(a), nameof(b));
        var result = new Matrix(a.Rows, a.Cols);
        for (int i = 0; i < a.Data.Length; i++)
        {
            result.Data[i] = a.Data[i] * b.Data[i];
        }
        return result;
    }


    /// Matrix multiplication (dot product). Requires a.Cols == b.Rows.
    /// <param name="a">Left matrix (m x n).</param>
    /// <param name="b">Right matrix (n x p).</param>
    /// <returns>A new matrix of size (m x p).</returns>
    /// <exception cref="ArgumentException">Thrown if a.Cols != b.Rows.</exception>
    public static Matrix Dot(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
            throw new ArgumentException($"Matrix dimensions incompatible for multiplication: {a.Rows}x{a.Cols} * {b.Rows}x{b.Cols}. Left Cols must equal Right Rows.");

        var result = new Matrix(a.Rows, b.Cols);

        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Cols; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < a.Cols; k++)
                {
                    sum += a.Data[i * a.Cols + k] * b.Data[k * b.Cols + j];
                }
                result.Data[i * result.Cols + j] = sum;
            }
        }

        return result;
    }


    /// Returns a new matrix that is the transpose of this matrix.
    /// <returns>A new matrix with dimensions (Cols x Rows).</returns>
    public Matrix Transpose()
    {
        var result = new Matrix(Cols, Rows);
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
            {
                result.Data[c * result.Cols + r] = Data[r * Cols + c];
            }
        }
        return result;
    }


    /// Applies a function element-wise and returns a new matrix with the results.
    /// <param name="func">Function to apply to each element.</param>
    /// <returns>A new matrix with func applied to each element.</returns>
    /// <exception cref="ArgumentNullException">Thrown if func is null.</exception>
    public Matrix Apply(Func<double, double> func)
    {
        if (func == null)
            throw new ArgumentNullException(nameof(func));

        var result = new Matrix(Rows, Cols);
        for (int i = 0; i < Data.Length; i++)
        {
            result.Data[i] = func(Data[i]);
        }
        return result;
    }


    /// Applies a function element-wise in place (mutates this matrix).
    /// <param name="func">Function to apply to each element.</param>
    /// <exception cref="ArgumentNullException">Thrown if func is null.</exception>
    public void ApplyInPlace(Func<double, double> func)
    {
        if (func == null)
            throw new ArgumentNullException(nameof(func));

        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] = func(Data[i]);
        }
    }


    /// Returns a copy of the specified row as an array.
    /// <param name="row">Row index (0-based).</param>
    /// <returns>A new array containing the row data.</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown if row is out of bounds.</exception>
    public double[] GetRow(int row)
    {
        if (row < 0 || row >= Rows)
            throw new IndexOutOfRangeException($"Row index {row} out of bounds for matrix with {Rows} rows");

        var result = new double[Cols];
        Array.Copy(Data, row * Cols, result, 0, Cols);
        return result;
    }


    /// Returns a copy of the specified column as an array.
    /// <param name="col">Column index (0-based).</param>
    /// <returns>A new array containing the column data.</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown if col is out of bounds.</exception>
    public double[] GetCol(int col)
    {
        if (col < 0 || col >= Cols)
            throw new IndexOutOfRangeException($"Column index {col} out of bounds for matrix with {Cols} columns");

        var result = new double[Rows];
        for (int r = 0; r < Rows; r++)
        {
            result[r] = Data[r * Cols + col];
        }
        return result;
    }


    /// Sets the values of a row from an array.
    /// <param name="row">Row index (0-based).</param>
    /// <param name="values">Array of values (must have length == Cols).</param>
    /// <exception cref="IndexOutOfRangeException">Thrown if row is out of bounds.</exception>
    /// <exception cref="ArgumentException">Thrown if values length != Cols.</exception>
    public void SetRow(int row, double[] values)
    {
        if (row < 0 || row >= Rows)
            throw new IndexOutOfRangeException($"Row index {row} out of bounds for matrix with {Rows} rows");
        if (values == null)
            throw new ArgumentNullException(nameof(values));
        if (values.Length != Cols)
            throw new ArgumentException($"Values length ({values.Length}) must equal Cols ({Cols})", nameof(values));

        Array.Copy(values, 0, Data, row * Cols, Cols);
    }


    /// Sets the values of a column from an array.
    /// <param name="col">Column index (0-based).</param>
    /// <param name="values">Array of values (must have length == Rows).</param>
    /// <exception cref="IndexOutOfRangeException">Thrown if col is out of bounds.</exception>
    /// <exception cref="ArgumentException">Thrown if values length != Rows.</exception>
    public void SetCol(int col, double[] values)
    {
        if (col < 0 || col >= Cols)
            throw new IndexOutOfRangeException($"Column index {col} out of bounds for matrix with {Cols} columns");
        if (values == null)
            throw new ArgumentNullException(nameof(values));
        if (values.Length != Rows)
            throw new ArgumentException($"Values length ({values.Length}) must equal Rows ({Rows})", nameof(values));

        for (int r = 0; r < Rows; r++)
        {
            Data[r * Cols + col] = values[r];
        }
    }


    /// Converts the matrix to a jagged array (array of row arrays) for debugging/inspection.
    /// <returns>A jagged array where each inner array is a row.</returns>
    public double[][] ToJagged()
    {
        var result = new double[Rows][];
        for (int r = 0; r < Rows; r++)
        {
            result[r] = new double[Cols];
            Array.Copy(Data, r * Cols, result[r], 0, Cols);
        }
        return result;
    }


    /// Returns a string representation of the matrix for debugging.
    /// <returns>A formatted string showing the matrix dimensions and values.</returns>
    public override string ToString()
    {
        if (Rows == 0 || Cols == 0)
            return $"Matrix[{Rows}x{Cols}] (empty)";

        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Matrix[{Rows}x{Cols}]:");
        for (int r = 0; r < Rows; r++)
        {
            sb.Append("  [");
            for (int c = 0; c < Cols; c++)
            {
                sb.Append(Data[r * Cols + c].ToString("G6"));
                if (c < Cols - 1) sb.Append(", ");
            }
            sb.AppendLine("]");
        }
        return sb.ToString();
    }


    /// Creates a matrix filled with zeros.
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <returns>A new zero matrix.</returns>
    public static Matrix Zeros(int rows, int cols) => new Matrix(rows, cols, 0.0);


    /// Creates a matrix filled with ones.
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <returns>A new matrix filled with 1.0.</returns>
    public static Matrix Ones(int rows, int cols) => new Matrix(rows, cols, 1.0);


    /// Creates a matrix with random values uniformly distributed in [min, max).
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="min">Minimum value (inclusive).</param>
    /// <param name="max">Maximum value (exclusive).</param>
    /// <param name="rng">Optional Random instance. If null, a new one is created.</param>
    /// <returns>A new matrix with random values.</returns>
    public static Matrix Random(int rows, int cols, double min, double max, Random? rng = null)
    {
        rng ??= new Random();
        var result = new Matrix(rows, cols);
        for (int i = 0; i < result.Data.Length; i++)
        {
            result.Data[i] = rng.NextDouble() * (max - min) + min;
        }
        return result;
    }


    /// Creates an identity matrix of the given size.
    /// <param name="size">Size of the square matrix.</param>
    /// <returns>A square identity matrix.</returns>
    /// <exception cref="ArgumentException">Thrown if size is negative.</exception>
    public static Matrix Identity(int size)
    {
        if (size < 0)
            throw new ArgumentException($"Size must be non-negative, got {size}", nameof(size));

        var result = new Matrix(size, size);
        for (int i = 0; i < size; i++)
        {
            result.Data[i * size + i] = 1.0;
        }
        return result;
    }


    /// Validates that two matrices have the same dimensions.
    private static void ValidateSameShape(Matrix a, Matrix b, string paramA, string paramB)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols)
        {
            throw new ArgumentException(
                $"Matrix dimensions must match for element-wise operation: {paramA} is {a.Rows}x{a.Cols}, {paramB} is {b.Rows}x{b.Cols}");
        }
    }
}