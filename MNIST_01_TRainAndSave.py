import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense_layer import Dense
from convolution_layer import Convolution
from activation_function import Sigmoid
from loss import binary_cross_entropy, binary_cross_entropy_prime
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

def save_model(network, filename):
    model_parameters = {}
    for i, layer in enumerate(network):
        if hasattr(layer, 'weights'):
            model_parameters[f'layer_{i}_weights'] = layer.weights
        if hasattr(layer, 'bias'):
            model_parameters[f'layer_{i}_bias'] = layer.bias
            
    np.save(filename, model_parameters)

# 載入 MNIST 數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
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

epochs = 20
learning_rate = 0.1

# 訓練
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # 前向傳播
        output = x
        for layer in network:
            output = layer.forward(output)
        
        # 計算誤差
        error += binary_cross_entropy(y, output)
        
        # 反向傳播
        gradient = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)

    error /= len(x_train)
    print(f"{e + 1} / {epochs}, error = {error}")

# 儲存模型參數
save_model(network, 'model_parameters.npy')
