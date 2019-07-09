from dataset.mnist import load_mnist
from functions import *
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import pickle
import time

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)
print('== Shape of Training & Test set ==')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print('== type & shape of the data ==')
print(x_train[0].shape, type(x_train[0]))
print(y_train[0].shape, type(y_train[0]))
print(x_test[0].shape, type(x_test[0]))
print(y_test[0].shape, type(y_test[0]))

image = x_train[0]
label = y_train[0]
print(label)
imshow(image.reshape(28, 28))
plt.title('Sample data')
plt.show()

f = open('./MNIST_1_Weights.pkl', 'rb') #read in binary
network = pickle.load(f)


def predict(X):
    keys = sorted(network.keys())
    W1, W2, W3 = (network[idx] for idx in keys[:3])
    b1, b2, b3 = (network[idx] for idx in keys[3:])

    a1 = np.dot(X, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    return softmax(a3)


import time

accuracy = 0
wrong_idx = list()
batch_size = 100

start_time = time.time()
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(x_batch)
    t_batch = y_test[i:i+batch_size]
    accuracy += np.sum(y_batch == t_batch)
    wrong = y_batch != t_batch
    for i in range(wrong):
        if wrong[i]:
            wrong_idx.append(i)

now = time.time()
print('{} seconds spent predicting {} cases'.format(now - start_time, len(x_test)))
print('Accuracy: {}'.format(float(accuracy / len(x_test))))

#Show test cases that my network made right answer
from random import randint
idx = randint(0, len(x_test)-1)
y = predict(x_test[idx])
imshow(x_test[idx].reshape(28,28))
plt.title('Cases made right answer : {}'.format(np.argmax(y)))
plt.show()

#Show test cases that my network made wrong answer
idx = wrong_idx[randint(0, len(wrong_idx)-1)]
y = predict(x_test[idx])
imshow(x_test[idx].reshape(28, 28))
plt.title('Cases made wrong answer : {}'.format(np.argmax(y)))
plt.show()