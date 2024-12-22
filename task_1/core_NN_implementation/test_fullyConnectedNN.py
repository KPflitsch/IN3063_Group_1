import unittest
import numpy as np

from task_1.core_NN_implementation.fullyConnectedNN import fullyConnectedNN
from cifar10_loader import load_cifar10

# Unit tests
class TestFullyConnectedNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load CIFAR-10
        data_dir = "path_to_cifar10_directory"  # Replace with the actual dataset path
        train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)
        cls.train_data, cls.train_labels, cls.test_data, cls.test_labels = preprocess_cifar10(
            train_data, train_labels, test_data, test_labels
        )

        # Initialize the neural network
        cls.nn = fullyConnectedNN(
            input_size=3072,         # CIFAR-10 input size
            output_size=10,          # 10 classes
            hidden_layers=[128, 64], # Example configuration
            learning_rate=0.01,
            dropout_rate=0.2,
            regularization_rate=0.01
        )

    def test_initialization(self):
        """Test that weights and biases are initialized with correct shapes."""
        for i in range(len(self.nn.weights)):
            input_dim = self.nn.layer_sizes[i]
            output_dim = self.nn.layer_sizes[i + 1]
            self.assertEqual(self.nn.weights[i].shape, (input_dim, output_dim))
            self.assertEqual(self.nn.biases[i].shape, (1, output_dim))

    def test_forward_propagation(self):
        """Test forward propagation produces correct output shapes."""
        output = self.nn.forward_propagation(self.train_data[:10])  # Test on a small batch
        self.assertEqual(output.shape, (10, 10))  # Output shape should match (batch_size, num_classes)

    def test_loss_calculation(self):
        """Test loss calculation for valid values."""
        predictions = self.nn.forward_propagation(self.train_data[:10])
        loss = self.nn.calculate_loss(self.train_labels[:10], predictions)
        self.assertGreaterEqual(loss, 0)  # Loss should never be negative

    def test_training(self):
        """Test that training reduces the loss."""
        initial_loss = self.nn.calculate_loss(
            self.train_labels[:10],
            self.nn.forward_propagation(self.train_data[:10])
        )
        self.nn.train(self.train_data[:10], self.train_labels[:10], epochs=5)
        final_loss = self.nn.calculate_loss(
            self.train_labels[:10],
            self.nn.forward_propagation(self.train_data[:10])
        )
        self.assertLess(final_loss, initial_loss)  # Loss should decrease after training

    def test_predictions(self):
        """Test predictions are within valid range."""
        predictions = self.nn.predict(self.test_data[:10])
        for pred in predictions:
            self.assertIn(pred, range(10))  # Predictions should be class indices (0-9)

if __name__ == "__main__":
    unittest.main()