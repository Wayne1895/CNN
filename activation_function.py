from layer import Layer
from activation_layer import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2
        
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
#chatgpt
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_prime)

class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            e_x = np.exp(x - np.max(x))  # 防止溢出
            return e_x / np.sum(e_x, axis=0, keepdims=True)

        def softmax_prime(x):
            # Softmax的導數較為複雜，這裡可以直接返回一個簡單的示例
            return x * (1 - x)

        super().__init__(softmax, softmax_prime)