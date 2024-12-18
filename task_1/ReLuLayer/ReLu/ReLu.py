import numpy as np

class ReLuLayer:
    def __init__(self):
        self.cache=None

    def forward(self,x):
        self.cache=x
        return np.maximum(0,x)


    def backward(self,dout):
        dx=dout.copy()
        dx[self.cache<=0]=0
        return dx
