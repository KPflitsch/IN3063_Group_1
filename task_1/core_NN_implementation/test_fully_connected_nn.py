import unittest
import numpy as np
from task_1.ReLuLayer.ReLu.ReLu import ReLuLayer
from task_1.Sigmoid.Sigmoid.Sigmoid import SigmoidLayer
from task_1.dropout.dropout import Dropout
from task_1.softmaxLayer import SoftmaxLayer
from task_1.core_NN_implementation.fullyConnetedNN2 import fullyConnectedNN
from task_1.dataset.datasetLoader import load_cifar10

# Test class for the fullyConnectedNN model
class TestFullyConnectedNN(unittest.TestCase):

    # Setup CIFAR-10 dataset and model
    def setUp(self):
        # Load and preprocess the CIFAR-10 dataset
        data_dir = "task_1/dataset/cifar-10-batches-py"
        X_train, y_train, X_test, y_test = load_cifar10(data_dir)
        X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten and normalize
        X_test = X_test.reshape(X_test.shape[0], -1) / 255.0  # Flatten and normalize
        y_train_one_hot = np.eye(10)[y_train]  # One-hot encode labels
        y_test_one_hot = np.eye(10)[y_test]  # One-hot encode labels

        self.X_train = X_train
        self.y_train = y_train_one_hot
        self.X_test = X_test
        self.y_test = y_test_one_hot

        # Initialize a simple model for testing
        self.model = fullyConnectedNN(
            input_size=X_train.shape[1], 
            output_size=10, 
            hidden_layers=[128],
            activations=['relu', 'softmax'],
            dropout_rate=0.5,
            regularization='L2'
        )

    # Test forward propagation (ensure it doesn't crash)
    def test_forward_propagation(self):
        predictions = self.model.forward_propagation(self.X_train[:10], training=True)
        self.assertEqual(predictions.shape, (10, 10))  # Should output 10 predictions (one per sample) with 10 classes

    # Test loss calculation (ensure no NaN or Inf)
    def test_loss_calculation(self):
        predictions = self.model.forward_propagation(self.X_train[:10], training=True)
        loss = self.model.calculate_loss(self.y_train[:10], predictions)
        self.assertFalse(np.isnan(loss))  # Ensure loss is not NaN
        self.assertGreater(loss, 0)  # Loss should be positive

    # Test backward propagation (ensure it doesn't crash)
    def test_backward_propagation(self):
        self.model.backward_propagation(self.X_train[:10], self.y_train[:10])

    # Test the model's accuracy on a small batch
    def test_accuracy(self):
        predictions = self.model.predict(self.X_train[:10])
        accuracy = np.mean(predictions == np.argmax(self.y_train[:10], axis=1))
        self.assertGreaterEqual(accuracy, 0)  # Accuracy should be at least 0

    # Test training method (ensure it runs without errors)
    def test_train_method(self):
        history = self.model.train(self.X_train[:100], self.y_train[:100], epochs=1, batch_size=16)
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
    
    # Test if dropout is applied during training (ensure there's a difference between training and inference)
    def test_dropout(self):
        # Forward pass during training (dropout applied)
        train_predictions = self.model.forward_propagation(self.X_train[:10], training=True)
        train_output = train_predictions

        # Forward pass during inference (dropout should be turned off)
        inference_predictions = self.model.forward_propagation(self.X_train[:10], training=False)
        inference_output = inference_predictions

        # Assert that the outputs are different due to dropout
        self.assertFalse(np.allclose(train_output, inference_output))

    # Test regularization effect (L2 regularization)
    def test_l2_regularization(self):
        initial_weights = self.model.weights[0].copy()  # Save initial weights
        predictions = self.model.forward_propagation(self.X_train[:10], training=True)
        initial_loss = self.model.calculate_loss(self.y_train[:10], predictions)

        # Perform a step of gradient descent (backpropagation) to change weights
        self.model.backward_propagation(self.X_train[:10], self.y_train[:10])
        predictions_after = self.model.forward_propagation(self.X_train[:10], training=True)
        final_loss = self.model.calculate_loss(self.y_train[:10], predictions_after)

        # Check if the loss decreased (due to weight updates), which indicates L2 regularization working
        self.assertLess(final_loss, initial_loss)

    # Test model initialization (weights and biases should be initialized)
    def test_initialization(self):
        self.assertEqual(self.model.weights[0].shape, (self.X_train.shape[1], 128))  # Check first layer size
        self.assertEqual(self.model.biases[0].shape, (1, 128))  # Check first layer bias size
        self.assertTrue(np.any(self.model.weights[0] != 0))  # Ensure weights are initialized (not all zeros)
        self.assertTrue(np.any(self.model.biases[0] != 0))  # Ensure biases are initialized (not all zeros)

if __name__ == "__main__":
    unittest.main()