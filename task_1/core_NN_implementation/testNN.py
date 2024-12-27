import unittest
import numpy as np
from task_1.core_NN_implementation.fullyConnetedNN2 import fullyConnectedNN

class TestFullyConnectedNN(unittest.TestCase):

    def setUp(self):
        #Set up a simple neural network with 2 input features, 3 hidden units, and 2 output classes.
        self.nn = fullyConnectedNN(input_size=2, output_size=2, hidden_layers=[3], 
                                   activations=['relu', 'relu', 'softmax'], learning_rate=0.001, 
                                   dropout_rate=0.5, regularization='L2')
        # Dummy data for testing
        self.X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 3 samples, 2 features
        self.y = np.array([[1, 0], [0, 1], [1, 0]])  # One-hot encoded labels for 2 classes

    def test_forward_propagation_output_shape(self):
        #Test that forward propagation gives the correct output shape.
        predictions = self.nn.forward_propagation(self.X)
        self.assertEqual(predictions.shape, (3, 2), "Output shape should be (3, 2)")

    def test_loss_calculation(self):
        #Test that the loss function returns a valid loss.
        y_pred = self.nn.forward_propagation(self.X)
        loss = self.nn.calculate_loss(self.y, y_pred)
        self.assertGreaterEqual(loss, 0, "Loss should be non-negative")

    def test_backward_propagation(self):
        #Test that backward propagation updates the weights and biases.
        initial_weights = [w.copy() for w in self.nn.weights]
        initial_biases = [b.copy() for b in self.nn.biases]

        # Perform a single backward pass
        self.nn.backward_propagation(self.X, self.y)

        # Check that weights and biases are updated (i.e., changed)
        for i in range(len(self.nn.weights)):
            self.assertFalse(np.all(self.nn.weights[i] == initial_weights[i]), "Weights should be updated")
            self.assertFalse(np.all(self.nn.biases[i] == initial_biases[i]), "Biases should be updated")

    def test_training(self):
        #Test that training reduces the loss over time.
        initial_loss = self.nn.calculate_loss(self.y, self.nn.forward_propagation(self.X))

        # Train for a few epochs
        self.nn.train(self.X, self.y, epochs=5, batch_size=1)

        # Check that the loss decreases
        final_loss = self.nn.calculate_loss(self.y, self.nn.forward_propagation(self.X))
        self.assertLess(final_loss, initial_loss, "Loss should decrease after training")

    def test_prediction(self):
        #Test that the model produces valid predictions.
        predictions = self.nn.predict(self.X)
        self.assertEqual(predictions.shape, (3,), "Prediction output should be a vector of shape (3,)")

        # Ensure predictions are among the class indices (0 or 1 for a 2-class problem)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])), "Predictions should be in the set {0, 1}")

    def test_evaluation(self):
        #Test that evaluation computes loss and accuracy correctly.
        loss, accuracy = self.nn.evaluate(self.X, self.y)
        self.assertGreaterEqual(loss, 0, "Loss should be non-negative")
        self.assertGreaterEqual(accuracy, 0, "Accuracy should be between 0 and 1")
        self.assertLessEqual(accuracy, 1, "Accuracy should be between 0 and 1")

    def test_regularization(self):
        #Test that L2 regularization is applied during loss calculation.
        
        # Get loss without regularization
        original_loss = self.nn.calculate_loss(self.y, self.nn.forward_propagation(self.X))
        
        # Change regularization to L1
        self.nn.regularization = 'L1'
        l1_loss = self.nn.calculate_loss(self.y, self.nn.forward_propagation(self.X))

        self.assertNotEqual(original_loss, l1_loss, "Loss should change with different regularization")

    def test_dropout(self):
        #Test that dropout changes the output during training.
        self.nn.dropout_rate = 0.5

        # Get forward pass output without dropout (training = False)
        self.nn.dropout.forward = lambda a, training=False: a  # No dropout applied during forward pass
        output_without_dropout = self.nn.forward_propagation(self.X, training=False)

        # Get forward pass output with dropout (training = True)
        output_with_dropout = self.nn.forward_propagation(self.X, training=True)

        # Check that the dropout output is different from the no-dropout output
        self.assertFalse(np.allclose(output_with_dropout, output_without_dropout), "Dropout should change the output during training")

    def test_weight_initialization(self):
        #Test that weights are initialized with small random values.
        initial_weights = self.nn.weights
        self.assertTrue(np.all(np.abs(initial_weights[0]) < 0.1), "Weights should be initialized with small values")

if __name__ == '__main__':
    unittest.main()