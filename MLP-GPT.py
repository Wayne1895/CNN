import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from keras.datasets import mnist

# Step 1: Load MNIST data
def load_data():
    (trainX, trainy), (testX, testy) = mnist.load_data()
    return (trainX, trainy), (testX, testy)

# Step 2: Preprocess data
def preprocess_data(trainX, testX):
    trainX = trainX.reshape(trainX.shape[0], -1) / 255.0
    testX = testX.reshape(testX.shape[0], -1) / 255.0
    return trainX, testX

# Step 3: Define the MLP and other necessary classes
class MLP:
    def __init__(self, din, dout):
        self.W = (2 * np.random.rand(dout, din) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        self.b = (2 * np.random.rand(dout) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        
    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, gradout):
        self.deltaW = gradout.T @ self.x
        self.deltab = gradout.sum(0)
        return gradout @ self.W

class SequentialNN:
    def __init__(self, blocks: list):
        self.blocks = blocks
        
    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self, gradout):
        for block in self.blocks[::-1]:
            gradout = block.backward(gradout)
        return gradout

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, gradout):
        new_grad = gradout.copy()
        new_grad[self.x < 0] = 0.
        return new_grad

class LogSoftmax:
    def forward(self, x):
        self.x = x
        return x - logsumexp(x, axis=1, keepdims=True)
    
    def backward(self, gradout):
        softmax = np.exp(self.x) / np.sum(np.exp(self.x), axis=1, keepdims=True)
        return gradout - softmax * gradout.sum(axis=1, keepdims=True)

class NLLLoss:
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        return -np.sum(pred[np.arange(pred.shape[0]), true])

    def backward(self):
        grad = np.zeros_like(self.pred)
        grad[np.arange(self.pred.shape[0]), self.true] = -1
        return grad
    
    def __call__(self, pred, true):
        return self.forward(pred, true)

class Optimizer:
    def __init__(self, lr, compound_nn: SequentialNN):
        self.lr = lr
        self.compound_nn = compound_nn
        
    def step(self):
        for block in self.compound_nn.blocks:
            if isinstance(block, MLP):
                block.W -= self.lr * block.deltaW
                block.b -= self.lr * block.deltab

def train(model, optimizer, trainX, trainy, loss_fct=NLLLoss(), nb_epochs=20, batch_size=100):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        batch_idx = np.random.choice(trainX.shape[0], batch_size, replace=False)
        x = trainX[batch_idx]
        target = trainy[batch_idx]

        prediction = model.forward(x)
        loss_value = loss_fct(prediction, target)
        training_loss.append(loss_value)
        gradout = loss_fct.backward()
        model.backward(gradout)
        optimizer.step()
    return training_loss

# Step 4: Train and evaluate the model
if __name__ == "__main__":
    (trainX, trainy), (testX, testy) = load_data()
    trainX, testX = preprocess_data(trainX, testX)

    input_dim = trainX.shape[1]
    output_dim = 10  # Number of classes (digits 0-9)

    mlp = SequentialNN([MLP(input_dim, 128), ReLU(), 
                        MLP(128, 64), ReLU(), 
                        MLP(64, output_dim), LogSoftmax()])
    optimizer = Optimizer(1e-3, mlp)

    training_loss = train(mlp, optimizer, trainX, trainy)

    # Compute test accuracy
    accuracy = 0
    for i in range(testX.shape[0]):
        prediction = mlp.forward(testX[i].reshape(1, -1)).argmax()
        if prediction == testy[i]: accuracy += 1
    print('Test accuracy', accuracy / testX.shape[0] * 100, '%')