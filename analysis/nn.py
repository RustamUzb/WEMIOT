import autograd.numpy as np
from autograd import grad


np.set_printoptions(precision=4)


def initialize():
    X = np.array([[0.05, 0.10]])  # Inputs
    W1 = np.array([[0.15, 0.20], [0.25, 0.30]])  # Weights to calculate outputs for hidden layer 1
    b1 = 0.35  # Bias for hidden layer 1
    W2 = np.array([[0.40, 0.45], [0.50, 0.55]])  # Weights to calculate outputs for output layer
    b2 = 0.60  # Bias for output layer
    Y = np.array([[0.01, 0.99]])  # Desired output
    learning_rate = 0.5
    no_of_iter = int(10000)
    return X, W1, b1, W2, b2, Y, learning_rate, no_of_iter


def forward_pass(X, W1, b1, W2, b2, Y):
    ### Forward pass: Calculate hidden layer 1 (there is only 1 hidden layer in this example)
    #Z1 = np.dot(X, W1.T) + b1  # WtX + b
    #A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid(z) = 1 / (1 + e^(-z))
    A1 = sigmoid(X, W1, b1)
    A2 = sigmoid(A1, W2, b2)

    h_x = grad(sigmoid, 1)
    h_x(X, W1, b1)
    print(A1)
    print(A2)
    ### Forward pass: Calculate output layer
    #Z2 = np.dot(A1, W2.T) + b2  # WtX + b
    #A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid(z) = 1 / (1 + e^(-z))
    ### Calculate error/cost function
    E = np.sum(1 / 2 * np.square(Y - A2))  # squared error function
    return A1, A2, E


def back_propagation(X, W1, W2, Y, A1, A2):
    ### Back propogation
    ### Adjust W2
    dEdA2 = A2 - Y
    dA2dZ2 = np.multiply(A2, 1 - A2)
    dZ2dW2 = A1
    dEdW2 = dEdA2 * dA2dZ2 * dZ2dW2
    W2_adj = W2 - 0.5 * dEdW2.T
    W2 = W2_adj
    ### Adjust W1
    dZ2dA1 = W2.T
    dA1dZ1 = np.multiply(A1, 1 - A1)
    dZ1dW1 = X
    dEdW1 = dEdA2 * dA2dZ2 * dZ2dA1 * dA1dZ1 * dZ1dW1
    W1_adj = W1 - 0.5 * dEdW1.T
    W1 = W1_adj
    return W1, W2

def sigmoid(x, w, b):
    return 1 / (1 + np.exp(-(np.dot(x, w.T) + b)))

def main():
    (X, W1, b1, W2, b2, Y, learning_rate, no_of_iter) = initialize()
    # for iter in range(0, no_of_iter):
    A1, A2, E = forward_pass(X, W1, b1, W2, b2, Y)
    W1, W2 = back_propagation(X, W1, W2, Y, A1, A2)
   # print(A2)
   # print('W1 = {} \n\n W2 = {} \n\n Output = {} \n Desired output = {} \n Error = {}'.format(W1, W2, A2, Y, E))


main()