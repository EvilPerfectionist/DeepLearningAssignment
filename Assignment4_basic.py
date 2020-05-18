import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def ind_to_char(ind, book_char):
    return book_char[np.argmax(ind)]

# def char_to_ind(char, book_char):
#     one_hot = np.zeros((1, len(book_char)))
#     one_hot[0][np.where(book_char == char)[0][0]] = 1
#     return one_hot

def char_to_ind(char, book_char):
    alphabet_size = len(book_char)
    ind = np.zeros((alphabet_size, 1), dtype=int)
    ind[book_char.index(char)] = 1
    return ind.T

class RNN:
    def __init__(self, K, book_char):
        self.m = 100
        self.eta = 0.1
        self.theta = 1e-8
        self.seq_length = 25
        self.epochs = 3
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

        self.m_b = np.zeros((self.m, 1))
        self.m_c = np.zeros((self.K, 1))

        self.m_U = np.zeros((self.m, self.K))
        self.m_W = np.zeros((self.m, self.m))
        self.m_V = np.zeros((self.K, self.m))


    def softmax(self, x):
    	""" Standard definition of the softmax function """
    	return np.exp(x) / np.sum(np.exp(x), axis=0)

    def ComputeCost(self, P, Y):
        l = 0.0
        for i in range(Y.shape[1]):
            y = Y[:, [i]]
            p = P[:, [i]]
            l += -np.log(np.dot(y.T, p))[0][0]
        #J = l / Y.shape[1]
        J = l
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

        self.grad_V = np.minimum(np.maximum(self.grad_V, -5), 5)
        self.grad_W = np.minimum(np.maximum(self.grad_W, -5), 5)
        self.grad_U = np.minimum(np.maximum(self.grad_U, -5), 5)
        self.grad_b = np.minimum(np.maximum(self.grad_b, -5), 5)
        self.grad_c = np.minimum(np.maximum(self.grad_c, -5), 5)

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

    def calculate_diff(self, grad_num, grad_aly):
        fraction_W = np.absolute(grad_num) + np.absolute(grad_aly)
        fraction_W[fraction_W < 1e-6] = 1e-6
        diff = np.sum(np.divide(np.absolute(grad_num - grad_aly), fraction_W))
        return diff

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

        print('Accumulative realtive error of gradient b: ' + str(self.calculate_diff(grad_b, self.grad_b)))
        print('Accumulative Realtive error of gradient c: ' + str(self.calculate_diff(grad_c, self.grad_c)))
        print('Accumulative Realtive error of gradient V: ' + str(self.calculate_diff(grad_V, self.grad_V)))
        print('Accumulative Realtive error of gradient U: ' + str(self.calculate_diff(grad_U, self.grad_U)))
        print('Accumulative Realtive error of gradient W: ' + str(self.calculate_diff(grad_W, self.grad_W)))

    def synthezise_text(self, x, h, len, b, c, W, U, V):
        Y = []
        for i in range(len):
            y = np.zeros((self.K, 1))
            h_list, a_list, P = self.forward_pass(x, h, b, c, W, U, V)
            label = np.random.choice(self.K, 1, p = P[:, 0])
            y[label] = 1
            Y.append(y)
            x = y

        return Y

    def synthezise_text2(self, x0, h0, n, b, c, W, U, V):
        Y = np.zeros((self.K, n))
        x = x0
        h = h0

        for i in range(n):
            h_list, a_list, P = self.forward_pass(x, h, b, c, W, U, V)
            label = np.random.choice(self.K, p=P[:, 0])

            Y[label][i] = 1
            x = np.zeros(x.shape)
            x[label] = 1

        return Y

    def forward_pass2(self, x, h, b, c, W, U, V):
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

    def compute_gradients2(self, P, X, Y, H, H0, A, V, W):
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

    def fit(self, book_data):

        n = len(book_data)
        nb_seq = ceil(float(n-1) / float(self.seq_length))
        smooth_loss = 0
        ite = 0
        losses = []

        for i in range(self.epochs):
            e = 0
            hprev = np.random.normal(0, 0.01, self.h0.shape)
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

                X = np.zeros((self.K, len(X_chars)), dtype=int)
                Y = np.zeros((self.K, len(X_chars)), dtype=int)

                for i in range(len(X_chars)):
                    X[:, i] = char_to_ind(X_chars[i], self.book_char)
                    Y[:, i] = char_to_ind(Y_chars[i], self.book_char)

                P, H1, A = self.forward_pass2(
                    X, hprev, self.b, self.c, self.W, self.U, self.V)

                H0 = np.zeros((self.m, len(X_chars)))
                H0[:, [0]] = self.h0
                H0[:, 1:] = H1[:, :-1]

                self.compute_gradients2(P, X, Y, H1, H0, A, self.V, self.W)

                loss = self.ComputeCost(P, Y)
                if smooth_loss !=0:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                else:
                    smooth_loss = loss

                self.m_b += np.multiply(self.grad_b, self.grad_b)
                self.m_c += np.multiply(self.grad_c, self.grad_c)
                self.m_U += np.multiply(self.grad_U, self.grad_U)
                self.m_W += np.multiply(self.grad_W, self.grad_W)
                self.m_V += np.multiply(self.grad_V, self.grad_V)

                self.b -= np.multiply(self.eta / np.sqrt(self.m_b + self.theta), self.grad_b)
                self.c -= np.multiply(self.eta / np.sqrt(self.m_c + self.theta), self.grad_c)
                self.U -= np.multiply(self.eta / np.sqrt(self.m_U + self.theta), self.grad_U)
                self.W -= np.multiply(self.eta / np.sqrt(self.m_W + self.theta), self.grad_W)
                self.V -= np.multiply(self.eta / np.sqrt(self.m_V + self.theta), self.grad_V)

                hprev = H1[:, [-1]]

                if ite % 100 == 0:
                    losses.append(smooth_loss)

                if ite % 1000 == 0:
                    print("ite:", ite, "smooth_loss:", smooth_loss)

                if ite % 10000 == 0:
                    Y_temp = self.synthezise_text2(X[:, [0]], hprev, 200, self.b, self.c, self.W, self.U, self.V)
                    string = ""
                    for i in range(Y_temp.shape[1]):
                        string += ind_to_char(Y_temp[:, [i]], self.book_char)

                    print(string)

                ite += 1

        Y_temp = self.synthezise_text(char_to_ind("H", self.char_list).T, self.h0, 1000, self.b, self.c, self.W, self.U, self.V)
        string = ""
        for i in range(Y_temp.shape[1]):
            string += ind_to_char(Y_temp[:, [i]], self.book_char)

        print(string)

    def training(self, book_data):
        loss_list = []
        smooth_loss = 0
        for epoch in range(self.epochs):
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
                    X_int[:, j] = char_to_ind(X_char[j], self.book_char)
                    Y_int[:, j] = char_to_ind(Y_char[j], self.book_char)
                h_list, a_list, P = self.forward_pass(X_int, hprev, self.b, self.c, self.W, self.U, self.V)
                self.back_prop(X_int, Y_int, h_list, a_list, P, self.V, self.W)
                hprev = h_list[:, [-1]]
                loss = self.ComputeCost(P, Y_int)

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
                    result_int = self.synthezise_text(X_int[:, [0]], hprev, 200, self.b, self.c, self.W, self.U, self.V)
                    result_char = ""
                    for k in range(len(result_int)):
                        result_char += ind_to_char(result_int[k], self.book_char)
                    print(result_char)

                    Y_temp = self.synthezise_text2(X_int[:, [0]], hprev, 200, self.b, self.c, self.W, self.U, self.V)
                    string = ""
                    for i in range(Y_temp.shape[1]):
                        string += ind_to_char(Y_temp[:, [i]], self.book_char)

                    print(string)

        loss_plot = plt.plot(loss_list, 'g', label="loss")
        plt.xlabel('update steps(per 100 iterations)')
        plt.ylabel('loss')
        plt.xticks(np.arange(0, len(loss_list), len(loss_list) // self.epochs))
        plt.legend()
        plt.savefig('loss.png')
        #plt.show()


book_data = ''
with open('/home/leon/DeepLearning/goblet_book.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    book_data += line

#book_char = np.unique(list(book_data))
book_char = []

for i in range(len(book_data)):
    if not(book_data[i] in book_char):
        book_char.append(book_data[i])
print(len(book_char))
print(book_char)
print(char_to_ind(' ', book_char))
K = len(book_char)
model = RNN(K, book_char)
#model.check_gradients(book_data[0: model.seq_length], book_data[1: model.seq_length + 1])
model.fit(book_data)
