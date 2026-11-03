namespace custom_mlp_backprop_cs.NN;


/// Provides activation functions and their derivatives for neural networks.
/// Includes both scalar and matrix (element-wise) versions.
public static class Activation
{
    // ============================================================
    // Scalar Versions
    // ============================================================

    
    /// Rectified Linear Unit (ReLU) activation: max(0, x).

    /// <param name="x">Input value.</param>
    /// <returns>x if x > 0, otherwise 0.</returns>
    public static double ReLU(double x) => x > 0 ? x : 0.0;

    
    /// Derivative of ReLU: 1 if x > 0, else 0.
    /// At x = 0, returns 0 (standard practice).

    /// <param name="x">Input value.</param>
    /// <returns>1.0 if x > 0, otherwise 0.0.</returns>
    public static double ReLUDerivative(double x) => x > 0 ? 1.0 : 0.0;

    
    /// Linear (identity) activation: returns the input unchanged.

    /// <param name="x">Input value.</param>
    /// <returns>The input value x.</returns>
    public static double Linear(double x) => x;

    
    /// Derivative of Linear activation: always 1.

    /// <param name="x">Input value (unused).</param>
    /// <returns>Always 1.0.</returns>
    public static double LinearDerivative(double x) => 1.0;

    // ============================================================
    // Matrix Versions (return new matrix)
    // ============================================================

    
    /// Applies ReLU activation element-wise to a matrix, returning a new matrix.

    /// <param name="m">Input matrix.</param>
    /// <returns>A new matrix with ReLU applied to each element.</returns>
    /// <exception cref="ArgumentNullException">Thrown if m is null.</exception>
    public static Matrix ReLU(Matrix m)
    {
        if (m == null)
            throw new ArgumentNullException(nameof(m));
        return m.Apply(ReLU);
    }

    
    /// Applies ReLU derivative element-wise to a matrix, returning a new matrix.

    /// <param name="m">Input matrix.</param>
    /// <returns>A new matrix with ReLU derivative applied to each element.</returns>
    /// <exception cref="ArgumentNullException">Thrown if m is null.</exception>
    public static Matrix ReLUDerivative(Matrix m)
    {
        if (m == null)
            throw new ArgumentNullException(nameof(m));
        return m.Apply(ReLUDerivative);
    }

    
    /// Applies Linear activation element-wise to a matrix, returning a new matrix.

    /// <param name="m">Input matrix.</param>
    /// <returns>A new matrix with Linear (identity) applied to each element.</returns>
    /// <exception cref="ArgumentNullException">Thrown if m is null.</exception>
    public static Matrix Linear(Matrix m)
    {
        if (m == null)
            throw new ArgumentNullException(nameof(m));
        return m.Apply(Linear);
    }

    
    /// Applies Linear derivative element-wise to a matrix, returning a new matrix.

    /// <param name="m">Input matrix.</param>
    /// <returns>A new matrix with Linear derivative (1.0) applied to each element.</returns>
    /// <exception cref="ArgumentNullException">Thrown if m is null.</exception>
    public static Matrix LinearDerivative(Matrix m)
    {
        if (m == null)
            throw new ArgumentNullException(nameof(m));
        return m.Apply(LinearDerivative);
    }

    // ============================================================
    // In-Place Matrix Versions (mutate the matrix)
    // ============================================================

    
    /// Applies ReLU activation element-wise in place (mutates the matrix).

    /// <param name="m">Matrix to mutate.</param>
    /// <exception cref="ArgumentNullException">Thrown if m is null.</exception>
    public static void ReLUInPlace(Matrix m)
    {
        if (m == null)
            throw new ArgumentNullException(nameof(m));
        m.ApplyInPlace(ReLU);
    }

    
    /// Applies ReLU derivative element-wise in place (mutates the matrix).

    /// <param name="m">Matrix to mutate.</param>
    /// <exception cref="ArgumentNullException">Thrown if m is null.</exception>
    public static void ReLUDerivativeInPlace(Matrix m)
    {
        if (m == null)
            throw new ArgumentNullException(nameof(m));
        m.ApplyInPlace(ReLUDerivative);
    }
}