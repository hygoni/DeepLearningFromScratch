from functions import *

''''
Premade Networks
'''

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    def predict(self, x):
        return np.dot(x, self.W)
    def loss(self, x, t):
        y = self.predict(x)
        y = softmax(y)
        return cross_entropy_error(y, t)


class TwoLayerNet:

    def __init__(self, input_s, output_s, hidden_s, std):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_s, hidden_s)
        self.params['b1'] = np.zeros(hidden_s)
        self.params['W2'] = std * np.random.randn(hidden_s, output_s)
        self.params['b2'] = np.zeros(output_s)

    def predict(self, X):
        a1 = np.dot(X, self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cee(y, t)

    def update(self, x, t, learning_rate=0.1):
        grad = {}
        for key in self.params:
            grad[key] = numerical_gradient_2d(lambda W: self.loss(x, t), self.params[key])
        for key in grad:
            self.params[key] -= learning_rate * grad[key]

    def accuracy(self, y, t):
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(y.shape[0])
        return accuracy