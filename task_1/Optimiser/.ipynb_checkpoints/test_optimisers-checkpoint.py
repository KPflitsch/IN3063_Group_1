import numpy as np
from optimiser import SGD, SGDWithMomentum

def test_sgd():
    print("Testing SGD Optimizer...")
    # Mock weights and gradients
    weights = {
        "layer1": {"W": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([1.0, 1.0])}
    }
    gradients = {
        "layer1": {"dW": np.array([[0.1, 0.2], [0.3, 0.4]]), "db": np.array([0.1, 0.1])}
    }

    # Initialize SGD optimizer
    optimizer = SGD(learning_rate=0.1)

    # Update weights
    updated_weights = optimizer.update(weights, gradients)

    # Print updated weights for verification
    print("Updated Weights:", updated_weights)

def test_sgd_with_momentum():
    print("\nTesting SGD with Momentum...")
    # Mock weights and gradients
    weights = {
        "layer1": {"W": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([1.0, 1.0])}
    }
    gradients = {
        "layer1": {"dW": np.array([[0.1, 0.2], [0.3, 0.4]]), "db": np.array([0.1, 0.1])}
    }

    # Initialize SGD with Momentum optimizer
    optimizer = SGDWithMomentum(learning_rate=0.1, momentum=0.9)

    # Perform multiple updates
    for i in range(5):
        weights = optimizer.update(weights, gradients)
        print(f"Iteration {i + 1}, Updated Weights:", weights)

if __name__ == "__main__":
    test_sgd()
    test_sgd_with_momentum()
