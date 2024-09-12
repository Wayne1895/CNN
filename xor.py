from dense_layer import Dense
from activation_function import Tanh
from loss import mse, mse_prime
import numpy as np

#測試用資料集
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
#

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

#
epochs = 10000
learning_rate = 0.1

#訓練
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        output = x
        # forward
        for layer in network:
            output = layer.forward(output)
            
        # error
        error += mse(y, output)
        
        # backward
        output_gradient =  mse_prime(y, output)
        for layer in reversed(network):
            output_gradient = layer.backward(output_gradient, learning_rate)
        
    error /= len(x)
    print('%d/%d, error = %f' %(e+1, epochs, error))