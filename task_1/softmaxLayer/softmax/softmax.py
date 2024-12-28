import numpy as np

class SoftmaxLayer:

    def __init__(self):
        pass

    def forward_pass(self, inputs):
        stabilized_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(stabilized_inputs)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def backward_pass(self, dvalues, output):
        batch_size = dvalues.shape[0]
        gradients = np.empty_like(dvalues)

        for i in range(batch_size):
            single_output = dvalues[i].reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            gradients[i] = np.dot(jacobian_matrix, dvalues[i])

        return gradients

