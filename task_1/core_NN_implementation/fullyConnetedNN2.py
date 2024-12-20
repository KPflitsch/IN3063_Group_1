import numpy as np

from dataset.datasetLoader import load_cifar10
from task_1.ReLuLayer.ReLu.ReLu import ReLuLayer
from task_1.Sigmoid.Sigmoid.Sigmoid import SigmoidLayer
from task_1.core_NN_implementation.fullyConnectedNN import fullyConnectedNN
from task_1.dropout.dropout import Dropout
from task_1.softmaxLayer import SoftmaxLayer



class fullyConnectedNN:
    def __init__(self, input_size, output_size, hidden_layers,activations=None,learning_rate = 0.001, dropout_rate = None, regularization = 'L2'):
        """
        Parameters:

        input_size (int): Number of input features
        output_size (int): Number of output neurons
        hidden_layers (list): List of hidden layer sizes
        learning_rate (float): Learning rate
        dropout_rate (float): Dropout rate
        regularization_rate (float): L2 regularization rate
        """

        # Init params
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.regularization = regularization
        self.regularization_rate = 0.01
        self.optimizer = 'sgd'

        # Init activation layers

        self.activation_layer = {
            'relu': ReLuLayer(),
            'sigmoid': SigmoidLayer(),
            'softmax': SoftmaxLayer(),
        }

        if activations is None:
            # Create list of layer objects
            self.activations = []
            # Add ReLU for each hidden layer
            for _ in range(len(hidden_layers)):
                self.activations.append('relu')
            # Add softmax for output layer
            self.activations.append('softmax')
        else:
            self.activations = activations

        # Init dropout
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = Dropout(rate=dropout_rate)
        else:
            self.dropout = None

        # Structure
        self.layer_sizes = [input_size] + hidden_layers + [output_size]

        # Init weights and biases
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
                        for i in range(len(self.layer_sizes) - 1)]
        self.biases = [np.zeros((1, self.layer_sizes[i + 1])) for i in range(len(self.layer_sizes) - 1)]

    def forward_propagation(self, X, training=True):
        inputs = X
        # Stores outputs to be used for backward pass
        self.layer_outputs = []

        for i in range(len(self.weights) - 1):
            # z = XW + b
            z = np.dot(inputs, self.weights[i]) + self.biases[i]

            # ReLU activation
            activation = self.activations[i]
            a = self.activation_layer[activation].forward(z)

            # Use dropout when training and if dropout is enabled
            if training and self.dropout is not None:
                a = self.dropout.forward(a)
                inputs = self.dropout.output
            else:
                inputs = a

            self.layer_outputs.append(inputs)

        # Softmax
        z_final = np.dot(inputs, self.weights[-1]) + self.biases[-1]
        predictions = self.activation_layer['softmax'].forward(z_final)

        # Softmax output
        self.layer_outputs.append(predictions)
        return predictions

    def backward_propagation(self, X, y):
        m = X.shape[0]  # Number of samples
        dz = self.layer_outputs[-1] - y  # Error gradient at output layer

        # Loop through the layers in reverse order
        for i in reversed(range(len(self.weights))):
            # Gradient with respect to weights
            if i > 0:
                dw = np.dot(self.layer_outputs[i - 1].T, dz) / m
            else:
                dw = np.dot(X.T, dz) / m

            # Gradient with respect to biases
            db = np.sum(dz, axis=0, keepdims=True) / m

            if self.regularization == 'L2':
                # L2 Regularization for weights
                dw += self.regularization_rate * self.weights[i]
            elif self.regularization == 'L1':
                # L1 Regularization for weights
                dw += self.regularization_rate * np.sign(self.weights[i])

            # Update weights and biases using SGD
            # Remove optimizer conditional since we're only using SGD
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

            # Propagate the error gradient to the previous layer
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.activation_derivative(
                    self.layer_outputs[i - 1], function='relu')

    def activation_derivative(self, activation_output, function='relu'):
        # ReLU: 1 = positive input, else 0
        if function == 'relu':
            return np.where(activation_output > 0, 1, 0)

        # Sigmoid: sig_out * (1 = sig_out)
        elif function == 'sigmoid':
            return activation_output * (1 - activation_output)

        else:
            raise ValueError(f"Invalid activation function. {function}")

    def calculate_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Use epsilon to avoid log(0)
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m

        #L1 and L2 loss
        if self.regularization == 'L2':
            regularization_loss = 0.5 * self.regularization_rate * sum(np.sum(w ** 2) for w in self.weights)
        else:
            regularization_loss = 0.5 * self.regularization_rate * sum(np.sum(w) for w in self.weights)
        return loss + regularization_loss

    def train(self, X, y, epochs, batch_size=64, X_val=None, y_val=None):
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                x_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Forward pass
                y_pred = self.forward_propagation(x_batch, training=True)

                # Loss
                loss = self.calculate_loss(y_batch, y_pred)
                epoch_loss += loss

                # Backward pass
                self.backward_propagation(x_batch, y_batch)

                num_batches += 1

            # Average loss for the epoch
            epoch_loss /= num_batches
            history['loss'].append(epoch_loss)

            # Compute training accuracy
            train_predictions = self.predict(X)
            train_accuracy = np.mean(train_predictions == np.argmax(y, axis=1))
            history['accuracy'].append(train_accuracy)

            # Compute validation loss and accuracy if validation data is provided
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
                      f"Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        return history

    def evaluate(self, X, y):
        # Predict and compute loss
        y_pred = self.forward_propagation(X, training=False)
        loss = self.calculate_loss(y, y_pred)

        # Accuracy
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

        return loss, accuracy

    def predict(self, X):
        probabilities = self.forward_propagation(X, training=False)
        return np.argmax(probabilities, axis=1)

