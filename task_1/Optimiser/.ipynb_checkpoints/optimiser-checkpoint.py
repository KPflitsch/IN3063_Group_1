# optimizers.py

import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        for layer in weights:
            weights[layer]['W'] -= self.learning_rate * gradients[layer]['dW']
            weights[layer]['b'] -= self.learning_rate * gradients[layer]['db']
        return weights


class SGDWithMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, weights, gradients):
        for layer in weights:
            if layer not in self.velocities:
                self.velocities[layer] = {
                    'dW': np.zeros_like(weights[layer]['W']),
                    'db': np.zeros_like(weights[layer]['b'])
                }
            
            self.velocities[layer]['dW'] = (
                self.momentum * self.velocities[layer]['dW'] +
                self.learning_rate * gradients[layer]['dW']
            )
            self.velocities[layer]['db'] = (
                self.momentum * self.velocities[layer]['db'] +
                self.learning_rate * gradients[layer]['db']
            )
            
            weights[layer]['W'] -= self.velocities[layer]['dW']
            weights[layer]['b'] -= self.velocities[layer]['db']
        
        return weights
