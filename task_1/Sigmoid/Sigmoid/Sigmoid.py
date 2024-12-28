import numpy as np

class SigmoidLayer:

    def __init__(self):
        self.cache=None

    def forward(self,y):
        output=1/(1+np.exp(-y))
        self.cache=output
        return output

    def backward(self,dout):
        sigmoid_value = self.cache
        derivative = sigmoid_value * (1.0 - sigmoid_value)
        return dout * derivative
