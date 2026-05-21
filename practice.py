import numpy as np

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    Y = np.zeros(points*classes, dtype= 'uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        Y[ix] = class_number
    return X, Y

import matplotlib as plt

print('here')
X, y = create_data(100, 3)
plt.scatter(X[:,0], X[:,1])
plt.show()


"""
class Layerdense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self,input):
        self.output = np.dot(input, self.weights) + self.biases

layer1 = Layerdense(4,5)
layer2 = Layerdense(5,3)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
"""