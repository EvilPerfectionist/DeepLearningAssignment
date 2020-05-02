import numpy as np
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, d, K, char_list):
        self.m = 100
        self.eta = 0.1
        self.seq_length = 25
        self.K = K

        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))

        self.mu, self.sigma = 0, 0.01
        self.U = np.random.normal(self.mu, self.sigma, (self.m, self.K))
        self.W = np.random.normal(self.mu, self.sigma, (self.m, self.m))
        self.V = np.random.normal(self.mu, self.sigma, (self.K, self.m))

    def softmax(self, x):
    	""" Standard definition of the softmax function """
    	return np.exp(x) / np.sum(np.exp(x), axis=0)

    def synthezise_text(self, x0, h0, n, b, c, W, U, V):
        x = x0
        ht = h0
        Y = np.zeros((self.K, n))
        for i in range(n):
            for t in range(x.shape[1]):
            a = np.dot(W, ht) + np.dot(U, x[:, [t]]) + b
            ht = np.tanh(a)
            o = np.dot(V, ht) + c
            p = self.softmax(o)
            label = np.random.choice(self.K, p)
            x = np.zeros(x.shape)
            x[label] = 1
            Y[label][i] = 1

        return Y

book_data = ''
with open('/home/leon/DeepLearning/goblet_book.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    book_data += line

book_char = np.unique(list(book_data))
K = len(book_char)
print(K)
