import numpy as np
import matplotlib.pyplot as plt

def ind_to_char(ind, book_char):
    return book_char[np.argmax(ind)]

def char_to_ind(char, book_char):
    one_hot = np.zeros((1, len(book_char)))
    one_hot[0][np.where(book_char == char)[0][0]] = 1
    return one_hot

class RNN:
    def __init__(self, K, book_char):
        self.m = 5
        self.eta = 0.1
        self.seq_length = 25
        self.K = K
        self.book_char = book_char
        self.h0 = np.zeros((self.m, 1))
        self.I_n = np.ones(self.seq_length).reshape(-1, 1)

        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))

        self.mu, self.sigma = 0, 0.01
        self.U = np.random.normal(self.mu, self.sigma, (self.m, self.K))
        self.W = np.random.normal(self.mu, self.sigma, (self.m, self.m))
        self.V = np.random.normal(self.mu, self.sigma, (self.K, self.m))

        self.grad_b = np.zeros((self.m, 1))
        self.grad_c = np.zeros((self.K, 1))

        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))

    def softmax(self, x):
    	""" Standard definition of the softmax function """
    	return np.exp(x) / np.sum(np.exp(x), axis=0)

    def ComputeCost(self, P, Y):
        l = 0.0
        for i in range(Y.shape[1]):
            y = Y[:, [i]]
            p = P[:, [i]]
            l += -np.log(np.dot(y.T, p))[0][0]
        J = l / Y.shape[1]
        return J

    def forward_pass(self, X, h, b, c, W, U, V):
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

        self.grad_V = np.where(self.grad_V < 5, self.grad_V, 5)
        self.grad_V = np.where(self.grad_V > -5, self.grad_V, -5)

        self.grad_W = np.where(self.grad_W < 5, self.grad_W, 5)
        self.grad_W = np.where(self.grad_W > -5, self.grad_W, -5)

        self.grad_U = np.where(self.grad_U < 5, self.grad_U, 5)
        self.grad_U = np.where(self.grad_U > -5, self.grad_U, -5)

        self.grad_b = np.where(self.grad_b<5, self.grad_b, 5)
        self.grad_b = np.where(self.grad_b>-5, self.grad_b, -5)

        self.grad_c = np.where(self.grad_c<5, self.grad_c, 5)
        self.grad_c = np.where(self.grad_c>-5, self.grad_c, -5)

    def ComputeGradientsNumSlow(self, X, Y, b, c, W, U, V):
        h = 1e-4

        grad_b = np.zeros((self.m, 1))
        grad_c = np.zeros((self.K, 1))
        grad_U = np.zeros((self.m, self.K))
        grad_W = np.zeros((self.m, self.m))
        grad_V = np.zeros((self.K, self.m))

        for i in range(b.shape[0]):
            b_try = np.copy(b)
            b_try[i] -= h

            h_list, a_list, P = self.forward_pass(X, self.h0, b_try, c, W, U, V)
            c1 = self.ComputeCost(P, Y)

            b_try = np.copy(b)
            b_try[i] += h

            h_list, a_list, P = self.forward_pass(X, self.h0, b_try, c, W, U, V)
            c2 = self.ComputeCost(P, Y)
            grad_b[i] = (c2 - c1) / (2 * h)

        for i in range(c.shape[0]):
            c_try = np.copy(c)
            c_try[i] -= h

            h_list, a_list, P = self.forward_pass(X, self.h0, b, c_try, W, U, V)
            c1 = self.ComputeCost(P, Y)

            c_try = np.copy(c)
            c_try[i] += h

            h_list, a_list, P = self.forward_pass(X, self.h0, b, c_try, W, U, V)
            c2 = self.ComputeCost(P, Y)
            grad_c[i] = (c2 - c1) / (2 * h)

        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V_try = np.copy(V)
                V_try[i][j] -= h

                h_list, a_list, P = self.forward_pass(X, self.h0, b, c, W, U, V_try)
                c1 = self.ComputeCost(P, Y)

                V_try = np.copy(V)
                V_try[i][j] += h

                h_list, a_list, P = self.forward_pass(X, self.h0, b, c, W, U, V_try)
                c2 = self.ComputeCost(P, Y)
                grad_V[i][j] = (c2 - c1) / (2 * h)


        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U_try = np.copy(U)
                U_try[i][j] -= h

                h_list, a_list, P = self.forward_pass(X, self.h0, b, c, W, U_try, V)
                c1 = self.ComputeCost(P, Y)

                U_try = np.copy(U)
                U_try[i][j] += h

                h_list, a_list, P = self.forward_pass(X, self.h0, b, c, W, U_try, V)
                c2 = self.ComputeCost(P, Y)
                grad_U[i][j] = (c2 - c1) / (2 * h)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i][j] -= h

                h_list, a_list, P = self.forward_pass(X, self.h0, b, c, W_try, U, V)
                c1 = self.ComputeCost(P, Y)

                W_try = np.copy(W)
                W_try[i][j] += h

                h_list, a_list, P = self.forward_pass(X, self.h0, b, c, W_try, U, V)
                c2 = self.ComputeCost(P, Y)
                grad_W[i][j] = (c2 - c1) / (2 * h)

        return grad_b, grad_c, grad_V, grad_U, grad_W

    def computeGradientsNumSlow2(self, X, Y, b, c, W, U, V):
        h = 1e-4

        grad_b = np.zeros((self.m, 1))
        grad_c = np.zeros((self.K, 1))
        grad_U = np.zeros((self.m, self.K))
        grad_W = np.zeros((self.m, self.m))
        grad_V = np.zeros((self.K, self.m))

        print("Computing b gradient")

        for i in range(b.shape[0]):
            b_try = np.copy(b)
            b_try[i] -= h

            P, _, _, = self.forward_pass(X, self.h0, b_try, c, W, U, V)
            c1 = self.computeCost(P, Y)

            b_try = np.copy(b)
            b_try[i] += h

            P, _, _, = self.forward_pass(X, self.h0, b_try, c, W, U, V)
            c2 = self.computeCost(P, Y)
            grad_b[i] = (c2 - c1) / (2 * h)

        print("Computing c gradient")

        for i in range(c.shape[0]):
            c_try = np.copy(c)
            c_try[i] -= h

            P, _, _, = self.forward_pass(X, self.h0, b, c_try, W, U, V)
            c1 = self.computeCost(P, Y)

            c_try = np.copy(c)
            c_try[i] += h

            P, _, _, = self.forward_pass(X, self.h0, b, c_try, W, U, V)
            c2 = self.computeCost(P, Y)
            grad_c[i] = (c2 - c1) / (2 * h)

        print("Computing V gradient")

        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V_try = np.copy(V)
                V_try[i][j] -= h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W, U, V_try)
                c1 = self.computeCost(P, Y)

                V_try = np.copy(V)
                V_try[i][j] += h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W, U, V_try)
                c2 = self.computeCost(P, Y)
                grad_V[i][j] = (c2 - c1) / (2 * h)

        print("Computing U gradient")

        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U_try = np.copy(U)
                U_try[i][j] -= h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W, U_try, V)
                c1 = self.computeCost(P, Y)

                U_try = np.copy(U)
                U_try[i][j] += h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W, U_try, V)
                c2 = self.computeCost(P, Y)
                grad_U[i][j] = (c2 - c1) / (2 * h)

        print("Computing W gradient")

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i][j] -= h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W_try, U, V)
                c1 = self.computeCost(P, Y)

                W_try = np.copy(W)
                W_try[i][j] += h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W_try, U, V)
                c2 = self.computeCost(P, Y)
                grad_W[i][j] = (c2 - c1) / (2 * h)

        return grad_b, grad_c, grad_V, grad_U, grad_W

    def check_gradients(self, X_char, Y_char):
        X_int = np.zeros((self.K, self.seq_length))
        Y_int = np.zeros((self.K, self.seq_length))

        for i in range(self.seq_length):
            X_int[:, i] = char_to_ind(X_char[i], self.book_char)
            Y_int[:, i] = char_to_ind(Y_char[i], self.book_char)

        h_list, a_list, P = self.forward_pass(X_int, self.h0, self.b, self.c, self.W, self.U, self.V)
        self.back_prop(X_int, Y_int, h_list, a_list, P, self.V, self.W)

        grad_b, grad_c, grad_V, grad_U, grad_W = self.ComputeGradientsNumSlow(
                X_int, Y_int, self.b, self.c, self.W, self.U, self.V)
        grad_b2, grad_c2, grad_V2, grad_U2, grad_W2 = self.ComputeGradientsNumSlow2(
                X_int, Y_int, self.b, self.c, self.W, self.U, self.V)

        print(sum(abs(grad_b - self.grad_b)) /
                  max(1e-4, sum(abs(grad_b)) + sum(abs(self.grad_b))))
        print(sum(abs(grad_c - self.grad_c)) /
              max(1e-4, sum(abs(grad_c)) + sum(abs(self.grad_c))))

        print(sum(sum(abs(grad_V - self.grad_V))) /
              max(1e-4, sum(sum(abs(grad_V))) + sum(sum(abs(self.grad_V)))))
        print(sum(sum(abs(grad_U - self.grad_U))) /
              max(1e-4, sum(sum(abs(grad_U))) + sum(sum(abs(self.grad_U)))))
        print(sum(sum(abs(grad_W - self.grad_W))) /
              max(1e-4, sum(sum(abs(grad_W))) + sum(sum(abs(self.grad_W)))))

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
print(book_char)
print(char_to_ind(' ', book_char))
K = len(book_char)
model = RNN(K, book_char)
model.check_gradients(book_data[0: model.seq_length], book_data[1: model.seq_length + 1])
