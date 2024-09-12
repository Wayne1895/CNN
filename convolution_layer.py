import numpy as np
from scipy import signal
from layer import Layer

class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_size = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_size)
        self.bias = np.random.randn(*self.output_shape)
    
    def forward(self, input):
        pass
        
    def backward(self, output_gradient, learning_rate):
        pass
        