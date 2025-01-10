import numpy as np

def hutchinson_trace_estimator(matrix_vector_product, n, num_samples=100):
    """
    Estimates the trace of matrix A using Hutchinson's trace estimator.
    
    Parameters:
    matrix_vector_product: callable
        Function that computes Ax given vector x
    n: int
        Dimension of the matrix
    num_samples: int
        Number of random vectors to use for estimation
    
    Returns:
    float: Estimated trace of the matrix
    """
    trace_estimates = []
    
    for _ in range(num_samples):
        # Generate random vector with ±1 entries (Rademacher distribution)
        v = np.random.choice([-1, 1], size=n)
        
        # Compute v^T * A * v
        Av = matrix_vector_product(v)
        estimate = np.dot(v, Av)
        
        trace_estimates.append(estimate)
    
    # Return mean of estimates
    return np.mean(trace_estimates)

# Example with a known matrix (just for demonstration)
def example():
    # Create a test matrix
    A = np.array([[1, 2], [3, 4]])
    
    # Define the matrix-vector product function
    def mv_product(x):
        return A @ x
    
    # Estimate the trace
    estimated_trace = hutchinson_trace_estimator(mv_product, n=len(A), num_samples=1000)
    
    print(f"True trace: {np.trace(A)}")
    print(f"Estimated trace: {estimated_trace}")

example()