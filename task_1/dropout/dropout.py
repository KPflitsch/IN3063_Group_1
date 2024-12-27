import numpy as np

class Dropout:
    
    def __init__(self,rate):
        
        if not (0 <= rate <= 1):
            raise ValueError("Dropout rate must be between 0 and 1.")
        self.rate = 1 - rate # Keep probability
        
    def forward (self, inputs, training=True):
        
        self.inputs = inputs
        if training:
            # Generate binary mask and scale it
            self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape)
            if self.rate == 0:
                self.output = 0
            else:
                self.output = inputs * self.binary_mask / self.rate
        else:
            # No dropout during inference
            self.output = inputs
        
    def backward (self, dvalues):

        # Gradient is passed only for active neurons
        self.dinputs = dvalues * self.binary_mask
        
        


