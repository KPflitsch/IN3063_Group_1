import unittest
import numpy as np
from dropout import Dropout


class TestDropout(unittest.TestCase):
    def test_inference(self):
        inputs = np.random.randn(10, 10)
        dropout = Dropout(rate=0.5)
        dropout.forward(inputs, training=False)
        self.assertTrue(np.allclose(dropout.output, inputs), "Inference mode failed!")
    
    def test_scaling(self):
        inputs = np.random.randn(1000, 1000)
        dropout = Dropout(rate=0.5)
        dropout.forward(inputs, training=True)
        self.assertAlmostEqual(np.mean(inputs), np.mean(dropout.output), delta=0.1, msg="Scaling failed!")

    def test_binary_mask(self):
        inputs = np.random.randn(100, 100)
        dropout = Dropout(rate=0.5)
        dropout.forward(inputs, training=True)
        active_neurons = np.sum(dropout.binary_mask) / np.prod(dropout.binary_mask.shape)
        self.assertAlmostEqual(active_neurons, dropout.rate, delta=0.05, msg="Binary mask proportion incorrect!")

if __name__ == "__main__":
    unittest.main()