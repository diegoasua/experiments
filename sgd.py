import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def loss_function(X):
    """
    Computes the loss: (1 - X^T * X)^2
    """
    product = np.trace(X.T @ X)
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

def loss_function_with_l2(X, lambda_l2):
    """
    Computes the loss with L2 regularization: (1 - X^T * X)^2 + lambda * ||X||^2
    """
    main_loss = loss_function(X)
    l2_reg = lambda_l2 * np.sum(X ** 2)
    return main_loss + l2_reg

def sgd_l2_optimizer(learning_rate=0.01, n_iterations=100, lambda_l2=0.01, momentum=0.9):
    X = np.random.randn(2, 2)
    velocity = np.zeros_like(X)
    loss_history = []
    
    for i in range(n_iterations):
        current_loss = loss_function_with_l2(X, lambda_l2)
        loss_history.append(current_loss)
        
        grad = gradient(X) + 2 * lambda_l2 * X  # Add L2 gradient
        velocity = momentum * velocity - learning_rate * grad
        X = X + velocity
        
        if (i + 1) % 10 == 0:
            print(f"SGD+L2 Iteration {i+1}: Loss = {current_loss:.6f}")
    
    return X, loss_history

def adam_l2_optimizer(learning_rate=0.01, n_iterations=100, lambda_l2=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    X = np.random.randn(2, 2)
    m = np.zeros_like(X)  # First moment
    v = np.zeros_like(X)  # Second moment
    loss_history = []
    
    for i in range(n_iterations):
        current_loss = loss_function_with_l2(X, lambda_l2)
        loss_history.append(current_loss)
        
        grad = gradient(X) + 2 * lambda_l2 * X  # Add L2 gradient
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        
        # Update X
        X = X - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        if (i + 1) % 10 == 0:
            print(f"Adam+L2 Iteration {i+1}: Loss = {current_loss:.6f}")
    
    return X, loss_history

def adamw_optimizer(learning_rate=0.01, n_iterations=100, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    X = np.random.randn(2, 2)
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    loss_history = []
    
    for i in range(n_iterations):
        current_loss = loss_function(X)  # Note: using original loss function
        loss_history.append(current_loss)
        
        grad = gradient(X)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        
        # AdamW update: separate weight decay
        X = X * (1 - learning_rate * weight_decay)
        X = X - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        if (i + 1) % 10 == 0:
            print(f"AdamW Iteration {i+1}: Loss = {current_loss:.6f}")
    
    return X, loss_history

# Compare optimizers with adjusted learning rates and regularization
lr_sgd = 0.01
lr_adam = 0.001  # Reduced learning rate for Adam
lr_adamw = 0.001  # Reduced learning rate for AdamW
n_iterations = 100
lambda_l2 = 0.001  # Reduced L2 regularization strength

# Run all optimizers
X_sgd_l2, loss_sgd_l2 = sgd_l2_optimizer(lr_sgd, n_iterations, lambda_l2)
X_adam_l2, loss_adam_l2 = adam_l2_optimizer(lr_adam, n_iterations, lambda_l2)
X_adamw, loss_adamw = adamw_optimizer(lr_adamw, n_iterations, lambda_l2)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(loss_sgd_l2, label='SGD+L2')
plt.plot(loss_adam_l2, label='Adam+L2')
plt.plot(loss_adamw, label='AdamW')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration - Optimizer Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Print final results
print("\nFinal Results:")
print("SGD+L2 final loss:", loss_sgd_l2[-1])
print("Adam+L2 final loss:", loss_adam_l2[-1])
print("AdamW final loss:", loss_adamw[-1])