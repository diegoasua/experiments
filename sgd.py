import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def loss_function(X):
    """
    Computes the loss: (1 - X^T * X)^2
    """
    product = np.trace(X.T @ X)  # Equivalent to x^T * x for matrix
    return (1 - product) ** 2

def gradient(X):
    """
    Computes the gradient of the loss with respect to X
    """
    product = np.trace(X.T @ X)
    # Derivative of (1 - x^T * x)^2 with respect to X
    return -4 * (1 - product) * X

def numerical_gradient(X, epsilon=1e-7):
    """
    Computes the gradient numerically using finite differences
    """
    grad = np.zeros_like(X)
    
    # Compute partial derivative for each element
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Create small perturbation matrix
            perturbation = np.zeros_like(X)
            perturbation[i, j] = epsilon
            
            # Central difference formula
            forward = loss_function(X + perturbation)
            backward = loss_function(X - perturbation)
            grad[i, j] = (forward - backward) / (2 * epsilon)
    
    return grad

def sgd_optimizer(learning_rate=0.01, n_iterations=100):
    # Initialize random 2x2 matrix
    X = np.random.randn(2, 2)
    
    # Lists to store loss values and X values for plotting
    loss_history = []
    X_history = []
    
    # SGD iterations
    for i in range(n_iterations):
        current_loss = loss_function(X)
        loss_history.append(current_loss)
        X_history.append(X.copy())
        
        grad = gradient(X)
        # Use numerical gradient instead of analytical gradient
        # grad = numerical_gradient(X)
        
        # Update X using gradient descent
        X = X - learning_rate * grad
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: Loss = {current_loss:.6f}")
    
    return X, loss_history, X_history

# Run optimization
final_X, loss_history, X_history = sgd_optimizer(learning_rate=0.01, n_iterations=100)

# Print final results
print("\nFinal X matrix:")
print(final_X)
print("\nFinal loss:", loss_function(final_X))

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration')
plt.grid(True)
plt.show()

# Verify the result
product = np.trace(final_X.T @ final_X)
print("\nX^T * X =", product)  # Should be close to 1 for optimal solution