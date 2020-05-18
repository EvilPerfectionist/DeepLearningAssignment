import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def char_to_ind(char, book_char):
    one_hot = np.zeros((1, len(book_char)))
    one_hot[0][np.where(book_char == char)[0][0]] = 1
    return one_hot

def ind_to_char(ind, book_char):
    return book_char[np.argmax(ind)]

class Vanilla_RNN:
    def __init__(self, d, K, char_list):
        self.m = 100
        self.eta = 0.1
        self.seq_length = 25
        self.d = d
        self.K = K
        self.epsilon = 1e-8
        self.theta = 1e-8
        self.I_n = np.ones(self.seq_length).reshape(-1, 1)

        self.nb_epochs = 4

        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))

        self.U = np.zeros((self.m, self.K))
        self.W = np.zeros((self.m, self.m))
        self.V = np.zeros((self.K, self.m))

        self.grad_b = np.zeros((self.m, 1))
        self.grad_c = np.zeros((self.K, 1))

        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))

        self.m_b = np.zeros((self.m, 1))
        self.m_c = np.zeros((self.K, 1))

        self.m_U = np.zeros((self.m, self.K))
        self.m_W = np.zeros((self.m, self.m))
        self.m_V = np.zeros((self.K, self.m))

        self.char_list = char_list

        self.h0 = np.zeros((self.m, 1))

        self.initialization()

    def initialization(self):
        mu = 0
        sigma = 0.01

        self.b = np.zeros(self.b.shape)
        self.c = np.zeros(self.c.shape)

        self.U = np.random.normal(mu, sigma, self.U.shape)
        self.W = np.random.normal(mu, sigma, self.W.shape)
        self.V = np.random.normal(mu, sigma, self.V.shape)

    def softmax(self, x):
        r = np.exp(x) / sum(np.exp(x))
        return r

    def forward_pass(self, x, h, b, c, W, U, V):
        ht = h
        H = np.zeros((self.m, x.shape[1]))
        P = np.zeros((self.K, x.shape[1]))
        A = np.zeros((self.m, x.shape[1]))
        for t in range(x.shape[1]):
            a = np.dot(W, ht) + np.dot(U, x[:, [t]]) + b
            ht = np.tanh(a)
            o = np.dot(V, ht) + c
            p = self.softmax(o)
            H[:, [t]] = ht
            P[:, [t]] = p
            A[:, [t]] = a
        return P, H, A

    def computeCost(self, P, Y):
        loss_sum = 0
        for i in range(P.shape[1]):
            p = P[:, [i]]
            y = Y[:, [i]]
            loss_sum += self.cross_entropy(p, y)
        assert(len(loss_sum) == 1)
        return loss_sum

    def cross_entropy(self, p, y):
        l = - np.log(np.dot(y.T, p))
        assert(len(l) == 1)
        return l

    def compute_gradients(self, P, X, Y, H, H0, A, V, W):
        G = -(Y.T - P.T).T

        self.grad_V = np.dot(G, H.T)
        self.grad_c = np.sum(G, axis=-1, keepdims=True)


        dLdh = np.zeros((X.shape[1], self.m))
        dLda = np.zeros((self.m, X.shape[1]))

        dLdh[-1] = np.dot(G.T[-1], V)
        dLda[:,-1] = np.multiply(dLdh[-1].T, (1 - np.multiply(np.tanh(A[:, -1]), np.tanh(A[:, -1]))))

        for t in range(X.shape[1]-2, -1, -1):
            dLdh[t] = np.dot(G.T[t], V) + np.dot(dLda[:, t+1], W)
            dLda[:,t] = np.multiply(dLdh[t].T, (1 - np.multiply(np.tanh(A[:, t]), np.tanh(A[:, t]))))

        self.grad_W = np.dot(dLda, H0.T)
        self.grad_U = np.dot(dLda, X.T)
        self.grad_b = np.sum(dLda, axis=-1, keepdims=True)

        self.grad_b = np.where(self.grad_b<5, self.grad_b, 5)
        self.grad_b = np.where(self.grad_b>-5, self.grad_b, -5)

        self.grad_c = np.where(self.grad_c<5, self.grad_c, 5)
        self.grad_c = np.where(self.grad_c>-5, self.grad_c, -5)

        self.grad_U = np.where(self.grad_U<5, self.grad_U, 5)
        self.grad_U = np.where(self.grad_U>-5, self.grad_U, -5)

        self.grad_V = np.where(self.grad_V<5, self.grad_V, 5)
        self.grad_V = np.where(self.grad_V>-5, self.grad_V, -5)

        self.grad_W = np.where(self.grad_W<5, self.grad_W, 5)
        self.grad_W = np.where(self.grad_W>-5, self.grad_W, -5)

    def synthezise_text(self, x0, h0, n, b, c, W, U, V):
        Y = np.zeros((self.K, n))
        x = x0
        h = h0

        for i in range(n):
            p, h, _ = self.forward_pass(x, h, b, c, W, U, V)
            label = np.random.choice(self.K, p=p[:, 0])

            Y[label][i] = 1
            x = np.zeros(x.shape)
            x[label] = 1

        return Y

    def fit(self, book_data):

        n = len(book_data)
        nb_seq = ceil(float(n-1) / float(self.seq_length))
        smooth_loss = 0
        ite = 0
        losses = []

        for i in range(self.nb_epochs):
            e = 0
            hprev = np.random.normal(0, 0.01, self.h0.shape)

            # if i != 0:
            #     self.eta /= 10

            print("epoch:", i)

            for j in range(nb_seq):

                if j == nb_seq-1:
                    X_chars = book_data[e:n - 2]
                    Y_chars = book_data[e + 1:n - 1]
                    e = n
                else:
                    X_chars = book_data[e:e + self.seq_length]
                    Y_chars = book_data[e + 1:e + self.seq_length + 1]
                    e += self.seq_length

                X = np.zeros((self.d, len(X_chars)), dtype=int)
                Y = np.zeros((self.K, len(X_chars)), dtype=int)

                for i in range(len(X_chars)):
                    X[:, i] = char_to_ind(X_chars[i], self.char_list)
                    Y[:, i] = char_to_ind(Y_chars[i], self.char_list)

                P, H1, A = self.forward_pass(
                    X, hprev, self.b, self.c, self.W, self.U, self.V)

                H0 = np.zeros((self.m, len(X_chars)))
                H0[:, [0]] = self.h0
                H0[:, 1:] = H1[:, :-1]

                self.compute_gradients(P, X, Y, H1, H0, A, self.V, self.W)

                loss = self.computeCost(P, Y)
                if smooth_loss !=0:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                else:
                    smooth_loss = loss

                self.m_b += np.multiply(self.grad_b, self.grad_b)
                self.m_c += np.multiply(self.grad_c, self.grad_c)
                self.m_U += np.multiply(self.grad_U, self.grad_U)
                self.m_W += np.multiply(self.grad_W, self.grad_W)
                self.m_V += np.multiply(self.grad_V, self.grad_V)

                self.b -= np.multiply(self.eta / np.sqrt(self.m_b + self.epsilon), self.grad_b)
                self.c -= np.multiply(self.eta / np.sqrt(self.m_c + self.epsilon), self.grad_c)
                self.U -= np.multiply(self.eta / np.sqrt(self.m_U + self.epsilon), self.grad_U)
                self.W -= np.multiply(self.eta / np.sqrt(self.m_W + self.epsilon), self.grad_W)
                self.V -= np.multiply(self.eta / np.sqrt(self.m_V + self.epsilon), self.grad_V)

                hprev = H1[:, [-1]]

                if ite % 100 == 0:
                    losses.append(smooth_loss)

                if ite % 1000 == 0:
                    print("ite:", ite, "smooth_loss:", smooth_loss)

                if ite % 10000 == 0:
                    Y_temp = self.synthezise_text(X[:, [0]], hprev, 200, self.b, self.c, self.W, self.U, self.V)
                    string = ""
                    for i in range(Y_temp.shape[1]):
                        string += ind_to_char(Y_temp[:, [i]], self.char_list)

                    print(string)

                ite += 1

    def forward_pass2(self, X, h, b, c, W, U, V):
        h_list = np.zeros((self.m, X.shape[1]))
        a_list = np.zeros((self.m, X.shape[1]))
        P = np.zeros((self.K, X.shape[1]))
        for i in range(X.shape[1]):
            a = np.dot(W, h) + np.dot(U, X[:, [i]]) + b
            h = np.tanh(a)
            o = np.dot(V, h) + c
            p = self.softmax(o)
            h_list[:, [i]] = h
            a_list[:, [i]] = a
            P[:, [i]] = p

        return h_list, a_list, P

    def back_prop(self, X, Y, h_list, a_list, P, V, W):
        G = -(Y - P)
        self.grad_V = np.dot(G, h_list.T)
        self.grad_c = np.dot(G, self.I_n)

        self.grad_h = np.zeros((X.shape[1], self.m))
        self.grad_h[-1, :] = np.dot(G.T[-1, :], V)

        self.grad_a = np.zeros((X.shape[1], self.m))
        self.grad_a[-1, :] = np.multiply(self.grad_h[-1, :], (1 - np.power(np.tanh(a_list[:, -1]), 2)).T)

        for i in range(X.shape[1] - 2, -1, -1):
            self.grad_h[i, :] = np.dot(G.T[i, :], V) + np.dot(self.grad_a[i + 1, :], W)
            self.grad_a[i, :] = np.multiply(self.grad_h[i, :], (1 - np.power(np.tanh(a_list[:, i]), 2)).T)

        h_list2 = np.zeros((self.m, X.shape[1]))
        h_list2[:, [0]] = self.h0
        h_list2[:, 1: X.shape[1]] = h_list[:, 0: X.shape[1] - 1]

        self.grad_W = np.dot(self.grad_a.T, h_list2.T)
        self.grad_U = np.dot(self.grad_a.T, X.T)
        self.grad_b = np.dot(self.grad_a.T, self.I_n)

        self.grad_V = np.minimum(np.maximum(self.grad_V, -5), 5)
        self.grad_W = np.minimum(np.maximum(self.grad_W, -5), 5)
        self.grad_U = np.minimum(np.maximum(self.grad_U, -5), 5)
        self.grad_b = np.minimum(np.maximum(self.grad_b, -5), 5)
        self.grad_c = np.minimum(np.maximum(self.grad_c, -5), 5)

    def training(self, book_data):
        loss_list = []
        smooth_loss = 0
        for epoch in range(self.nb_epochs):
            e = 0
            if epoch == 0:
                hprev = self.h0
            X_int = np.zeros((self.K, self.seq_length))
            Y_int = np.zeros((self.K, self.seq_length))
            for i in range(len(book_data) // self.seq_length + 1):
                if i == len(book_data) // self.seq_length:
                    X_char = book_data[e: len(book_data) - 1]
                    Y_char = book_data[e + 1: len(book_data)]
                else:
                    X_char = book_data[e: e + self.seq_length]
                    Y_char = book_data[e + 1: e + self.seq_length + 1]
                    e += self.seq_length
                for j in range(len(X_char)):
                    X_int[:, j] = char_to_ind(X_char[j], self.char_list)
                    Y_int[:, j] = char_to_ind(Y_char[j], self.char_list)
                h_list, a_list, P = self.forward_pass2(X_int, hprev, self.b, self.c, self.W, self.U, self.V)
                self.back_prop(X_int, Y_int, h_list, a_list, P, self.V, self.W)
                hprev = h_list[:, [-1]]
                loss = self.computeCost(P, Y_int)

                self.m_b += np.power(self.grad_b, 2)
                self.m_c += np.power(self.grad_c, 2)
                self.m_U += np.power(self.grad_U, 2)
                self.m_W += np.power(self.grad_W, 2)
                self.m_V += np.power(self.grad_V, 2)

                self.b -= np.multiply(self.eta / np.sqrt(self.m_b + self.theta), self.grad_b)
                self.c -= np.multiply(self.eta / np.sqrt(self.m_c + self.theta), self.grad_c)
                self.U -= np.multiply(self.eta / np.sqrt(self.m_U + self.theta), self.grad_U)
                self.W -= np.multiply(self.eta / np.sqrt(self.m_W + self.theta), self.grad_W)
                self.V -= np.multiply(self.eta / np.sqrt(self.m_V + self.theta), self.grad_V)

                if smooth_loss == 0:
                    smooth_loss = loss
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss

                if (i + 1) % 100 == 0:
                    loss_list.append(smooth_loss)
                    print(i, loss_list[-1])

                if i % 10000 == 0:

                    Y_temp = self.synthezise_text(X_int[:, [0]], hprev, 200, self.b, self.c, self.W, self.U, self.V)
                    string = ""
                    for i in range(Y_temp.shape[1]):
                        string += ind_to_char(Y_temp[:, [i]], self.char_list)

                    print(string)

book_data = ''
with open('/home/leon/DeepLearning/goblet_book.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    book_data += line

book_char = np.unique(list(book_data))
print(len(book_char))
print(book_char)
print(char_to_ind('H', book_char))
K = len(book_char)
net = Vanilla_RNN(K, K, book_char)
net.training(book_data)
#net.test_gradient(book_data[:net.seq_length], book_data[1:net.seq_length+1])
