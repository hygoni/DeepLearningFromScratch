from networks import *
from mnist import *
import matplotlib.pyplot as plt

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

for_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

net = TwoLayerNet(input_s=784, hidden_s=50, output_s=10, std=0.01)
loss_graph = list()
accuracy_graph = list()

sample = [i for i in range(1, 101)]
for i in range(10000):
    sample = np.random.choice(train_size, batch_size)
    x_batch = x_train[sample]
    t_batch = t_train[sample]
    y = net.predict(x_batch)

    loss_graph.append(net.loss(x_batch, t_batch))
    accuracy_graph.append(net.accuracy(y, t_batch))
    net.update(x_batch, t_batch, 0.01)
    if i % 5 == 0:
        print('Original Loss : {}'.format(loss_graph[0]))
    print('[Step {}] Loss : {}'.format(i, loss_graph[-1]))

x = np.arange(0, len(loss_graph), 1)
y = loss_graph
plt.plot(x, y)
plt.title('Loss Graph')
plt.show()

x = np.arange(0, len(accuracy_graph), 1)
y = accuracy_graph
plt.plot(x, y)
plt.title('Accuracy Graph (per Batch)')
plt.show()