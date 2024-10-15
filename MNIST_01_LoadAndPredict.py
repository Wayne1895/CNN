import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense_layer import Dense
from convolution_layer import Convolution
from activation_function import Sigmoid
from reshape_layer import Reshape

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(-1, 1, 28, 28)
    x = x.astype('float32') / 255
    y = np_utils.to_categorical(y, num_classes=2)
    y = y.reshape(-1, 2, 1)
    return x, y

def load_model(network, filename):
    model_parameters = np.load(filename, allow_pickle=True).item()
    for i, layer in enumerate(network):
        if hasattr(layer, 'weights'):
            layer.weights = model_parameters[f'layer_{i}_weights']
        if hasattr(layer, 'bias'):
            layer.bias = model_parameters[f'layer_{i}_bias']

# 載入 MNIST 數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test, y_test = preprocess_data(x_test, y_test, 100)

# 建立神經網絡
network = [
    Convolution((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),  # 最終輸出兩類
    Sigmoid()    
]

# 加載模型參數
load_model(network, 'model_parameters.npy')

# 測試
for x, y in zip(x_test, y_test):
    different = 0
    output = x
    for layer in network:
        output = layer.forward(output)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
    if np.argmax(output) !=  np.argmax(y):
        different += 1
print(f"different = {different}")