import unittest
import numpy as np

from task_1.core_NN_implementation.fullyConnectedNN import fullyConnectedNN
from cifar10_loader import load_cifar10

class TestFullyConnectedNNCIFAR10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load CIFAR-10 dataset
        data_dir = "path/to/cifar-10-batches-py"  # Update with the actual path
        train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)

        # Normalize and preprocess
        cls.X_train = train_data.reshape(train_data.shape[0], -1) / 255.0  # Flatten and normalize
        cls.X_test = test_data.reshape(test_data.shape[0], -1) / 255.0  # Flatten and normalize

        cls.y_train = np.eye(10)[train_labels]  # Convert to one-hot encoding
        cls.y_test = np.eye(10)[test_labels]  # Convert to one-hot encoding

        # Use a smaller subset for testing to save time
        cls.X_train_subset = cls.X_train[:1000]
        cls.y_train_subset = cls.y_train[:1000]
        cls.X_test_subset = cls.X_test[:200]
        cls.y_test_subset = cls.y_test[:200]

        # Initialize neural network
        cls.nn = fullyConnectedNN(
            input_size=3072,  # CIFAR-10 flattened image size
            output_size=10,   # 10 classes
            hidden_layers=[128, 64],  # Example hidden layer sizes
            learning_rate=0.01,
            dropout_rate=0.2,
            regularization='L2'
        )

    def test_cifar10_training(self):
        """Test training on CIFAR-10 subset."""
        initial_loss = self.nn.calculate_loss(
            self.y_train_subset,
            self.nn.forward_propagation(self.X_train_subset)
        )

        # Train the model on a small subset
        self.nn.train(self.X_train_subset, self.y_train_subset, epochs=5, batch_size=64)

        final_loss = self.nn.calculate_loss(
            self.y_train_subset,
            self.nn.forward_propagation(self.X_train_subset)
        )

        # Check if loss decreases
        self.assertLess(final_loss, initial_loss, "Training did not reduce the loss.")

    def test_cifar10_evaluation(self):
        """Test evaluation on CIFAR-10 test subset."""
        # Evaluate the model
        loss, accuracy = self.nn.evaluate(self.X_test_subset, self.y_test_subset)

        # Assert reasonable loss and accuracy values
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy must be non-negative.")
        self.assertLessEqual(accuracy, 1.0, "Accuracy must not exceed 1.")
        self.assertGreaterEqual(loss, 0.0, "Loss must be non-negative.")

    def test_cifar10_predictions(self):
        """Test predictions on CIFAR-10 test subset."""
        predictions = self.nn.predict(self.X_test_subset)

        # Check if predictions are valid class indices
        for pred in predictions:
            self.assertIn(pred, range(10), f"Prediction {pred} is out of range [0, 9].")


if __name__ == "__main__":
    unittest.main()