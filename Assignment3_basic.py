import numpy as np
import matplotlib.pyplot as plt
import math
import copy

dim = 3072
num_labs = 10
dims = [3072, 50, 50, 10]

def Initialization(dims, with_bn):
    W_list = []
    b_list = []
    if with_bn == True:
        G_list = []
        B_list = []

    for i in range(len(dims) - 1):
        mu, sigma = 0, 2 / math.sqrt(dims[i])
        W = np.random.normal(mu, sigma, (dims[i + 1], dims[i]))
        b = np.random.normal(mu, sigma, (dims[i + 1], 1))
        W_list.append(W)
        b_list.append(b)
    if with_bn == True:
        for j in range(len(dims) - 2):
            mu, sigma = 0, 2 / math.sqrt(dims[j])
            G = np.random.normal(mu, sigma, (dims[j + 1], 1))
            B = np.random.normal(mu, sigma, (dims[j + 1], 1))
            G_list.append(G)
            B_list.append(B)

    if with_bn == True:
        paras = {'W': W_list, 'b': b_list, 'G': G_list, 'B': B_list}
    else:
        paras = {'W': W_list, 'b': b_list}

    return paras

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open(filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def one_hot_representation(origin_labels):
    modified_labels = np.zeros((num_labs, len(origin_labels)))
    for index in range(len(origin_labels)):
        modified_labels[origin_labels[index]][index] = 1
    return modified_labels

def normal_representation(one_hot_labels):
    modified_labels = np.zeros(one_hot_labels.shape[1])
    for index in range(one_hot_labels.shape[1]):
        label = np.where(one_hot_labels[:, index] == 1)[0][0]
        modified_labels[index] = label
    return modified_labels

def Normalization(raw):
    theta = 1e-10
    mean = np.mean(raw, axis = 1)
    var = np.var(raw, axis = 1)
    norm = (raw - mean[:,None]) / np.power(var[:,None] + theta, 0.5)
    return norm

def EvaluateClassifier_BN(X, paras):
    for i in range(len(paras["W"]) - 1):
        S1 = np.dot(paras["W"][i], X) + paras["b"][i]
        S_norm = Normalization(S1)
        S_rescale = np.multiply(paras["G"][i], S_norm) + paras["B"][i]
        X = np.maximum(0, S_rescale)

    S = np.dot(paras["W"][-1], X) + paras["b"][-1]
    P = softmax(S)
    return P

def EvaluateClassifier(X, paras):
    for i in range(len(paras["W"]) - 1):
        S1 = np.dot(paras["W"][i], X) + paras["b"][i]
        X = np.maximum(0, S1)

    S = np.dot(paras["W"][-1], X) + paras["b"][-1]
    P = softmax(S)
    return P

def ComputeAccuracy(X, y, paras):
    p = EvaluateClassifier(X, paras)
    k = np.argmax(p, axis=0)
    acc = 1 - np.count_nonzero(k - y) / len(k)
    return acc

def ComputeAccuracy_Test(X, y, paras, mean_av_list, var_av_list):
    theta = 1e-10
    for i in range(len(paras["W"]) - 1):
        S1 = np.dot(paras["W"][i], X) + paras["b"][i]
        S_mean = mean_av_list[i]
        S_var = var_av_list[i]
        S_norm = (S1 - S_mean[:,None]) / np.power(S_var[:,None] + theta, 0.5)
        S_rescale = np.multiply(paras["G"][i], S_norm) + paras["B"][i]
        X = np.maximum(0, S_rescale)

    S = np.dot(paras["W"][-1], X) + paras["b"][-1]
    p = softmax(S)
    k = np.argmax(p, axis=0)
    acc = 1 - np.count_nonzero(k - y) / len(k)
    return acc

def ComputeLoss(X, Y, paras):
    P = EvaluateClassifier(X, paras)
    l = 0.0
    for i in range(Y.shape[1]):
        y = Y[:, [i]]
        p = P[:, [i]]
        l += -np.log(np.dot(y.T, p))[0][0]
    L = l / X.shape[1]
    return L

def ComputeCost(X, Y, paras, lamda):
    P = EvaluateClassifier(X, paras)
    l = 0.0
    for i in range(Y.shape[1]):
        y = Y[:, [i]]
        p = P[:, [i]]
        l += -np.log(np.dot(y.T, p))[0][0]
    reg = lamda * (np.sum(np.square(paras["W"][0])) + np.sum(np.square(paras["W"][1])))
    J = l / X.shape[1] + reg
    return J

def ComputeCost_BN(X, Y, paras, lamda):
    P = EvaluateClassifier_BN(X, paras)
    l = 0.0
    weight = 0.0
    for i in range(Y.shape[1]):
        y = Y[:, [i]]
        p = P[:, [i]]
        l += -np.log(np.dot(y.T, p))[0][0]
    for j in range(len(paras["W"])):
        weight += np.sum(np.square(paras["W"][j]))
    reg = lamda * weight
    J = l / X.shape[1] + reg
    return J

def ComputeCost_Test(X, Y, paras, lamda, mean_av_list, var_av_list):
    theta = 1e-10
    for i in range(len(paras["W"]) - 1):
        S1 = np.dot(paras["W"][i], X) + paras["b"][i]
        S_mean = mean_av_list[i]
        S_var = var_av_list[i]
        S_norm = (S1 - S_mean[:,None]) / np.power(S_var[:,None] + theta, 0.5)
        S_rescale = np.multiply(paras["G"][i], S_norm) + paras["B"][i]
        X = np.maximum(0, S_rescale)

    S = np.dot(paras["W"][-1], X) + paras["b"][-1]
    P = softmax(S)
    l = 0.0
    weight = 0.0
    for i in range(Y.shape[1]):
        y = Y[:, [i]]
        p = P[:, [i]]
        l += -np.log(np.dot(y.T, p))[0][0]
    for j in range(len(paras["W"])):
        weight += np.sum(np.square(paras["W"][j]))
    reg = lamda * weight
    J = l / X.shape[1] + reg
    return J

def ComputeGradsNum(X, Y, paras, lamda, h):

    grad_W_list = []
    grad_b_list = []

    c = ComputeCost(X, Y, paras, lamda);

    for i in range(len(paras["b"])):
        grad_b = np.zeros((len(paras["b"][i]), 1))
        paras_try = copy.deepcopy(paras)

        for j in range(len(paras["b"][i])):
            b_try = np.array(paras["b"][i])
            b_try[j] += h
            paras_try["b"][i] = b_try
            c2 = ComputeCost(X, Y, paras_try, lamda)
            grad_b[j] = (c2 - c) / h

        grad_b_list.append(grad_b)

    for k in range(len(paras["W"])):
        grad_W = np.zeros(paras["W"][k].shape)
        paras_try = copy.deepcopy(paras)

        for i in range(paras["W"][k].shape[0]):
            for j in range(paras["W"][k].shape[1]):
                W_try = np.array(paras["W"][k])
                W_try[i,j] += h
                paras_try["W"][k] = W_try
                c2 = ComputeCost(X, Y, paras_try, lamda)
                grad_W[i,j] = (c2-c) / h

        grad_W_list.append(grad_W)

    update_para = {'grad_W': grad_W_list, 'grad_b': grad_b_list}

    return update_para

def ComputeGradsNumSlow(X, Y, paras, lamda, h):

    grad_W_list = []
    grad_b_list = []

    for i in range(len(paras["b"])):
        grad_b = np.zeros((len(paras["b"][i]), 1))
        paras_try = copy.deepcopy(paras)

        for j in range(len(paras["b"][i])):
            b_try = np.array(paras["b"][i])
            b_try[j] -= h
            paras_try["b"][i] = b_try
            c1 = ComputeCost(X, Y, paras_try, lamda)

            b_try = np.array(paras["b"][i])
            b_try[j] += h
            paras_try["b"][i] = b_try
            c2 = ComputeCost(X, Y, paras_try, lamda)

            grad_b[j] = (c2 - c1) / (2 * h)

        grad_b_list.append(grad_b)

    for k in range(len(paras["W"])):
        grad_W = np.zeros(paras["W"][k].shape)
        paras_try = copy.deepcopy(paras)

        for i in range(paras["W"][k].shape[0]):
            for j in range(paras["W"][k].shape[1]):
                W_try = np.array(paras["W"][k])
                W_try[i,j] -= h
                paras_try["W"][k] = W_try
                c1 = ComputeCost(X, Y, paras_try, lamda)

                W_try = np.array(paras["W"][k])
                W_try[i,j] += h
                paras_try["W"][k] = W_try
                c2 = ComputeCost(X, Y, paras_try, lamda)

                grad_W[i,j] = (c2 - c1) / (2 * h)

        grad_W_list.append(grad_W)

    update_para = {'grad_W': grad_W_list, 'grad_b': grad_b_list}

    return update_para

def ComputeGradsNum_BN(X, Y, paras, lamda, h):

    grad_W_list = []
    grad_b_list = []
    grad_G_list = []
    grad_B_list = []

    for i in range(len(paras["b"])):
        grad_b = np.zeros((len(paras["b"][i]), 1))
        paras_try = copy.deepcopy(paras)

        for j in range(len(paras["b"][i])):
            b_try = np.array(paras["b"][i])
            b_try[j] -= h
            paras_try["b"][i] = b_try
            c1 = ComputeCost_BN(X, Y, paras_try, lamda)

            b_try = np.array(paras["b"][i])
            b_try[j] += h
            paras_try["b"][i] = b_try
            c2 = ComputeCost_BN(X, Y, paras_try, lamda)

            grad_b[j] = (c2 - c1) / (2 * h)

        grad_b_list.append(grad_b)

    for k in range(len(paras["W"])):
        grad_W = np.zeros(paras["W"][k].shape)
        paras_try = copy.deepcopy(paras)

        for i in range(paras["W"][k].shape[0]):
            for j in range(paras["W"][k].shape[1]):
                W_try = np.array(paras["W"][k])
                W_try[i,j] -= h
                paras_try["W"][k] = W_try
                c1 = ComputeCost_BN(X, Y, paras_try, lamda)

                W_try = np.array(paras["W"][k])
                W_try[i,j] += h
                paras_try["W"][k] = W_try
                c2 = ComputeCost_BN(X, Y, paras_try, lamda)

                grad_W[i,j] = (c2 - c1) / (2 * h)

        grad_W_list.append(grad_W)

    for i in range(len(paras["G"])):
        grad_G = np.zeros((len(paras["G"][i]), 1))
        paras_try = copy.deepcopy(paras)

        for j in range(len(paras["G"][i])):
            G_try = np.array(paras["G"][i])
            G_try[j] -= h
            paras_try["G"][i] = G_try
            c1 = ComputeCost_BN(X, Y, paras_try, lamda)

            G_try = np.array(paras["G"][i])
            G_try[j] += h
            paras_try["G"][i] = G_try
            c2 = ComputeCost_BN(X, Y, paras_try, lamda)

            grad_G[j] = (c2 - c1) / (2 * h)

        grad_G_list.append(grad_G)

    for i in range(len(paras["B"])):
        grad_B = np.zeros((len(paras["B"][i]), 1))
        paras_try = copy.deepcopy(paras)

        for j in range(len(paras["B"][i])):
            B_try = np.array(paras["B"][i])
            B_try[j] -= h
            paras_try["B"][i] = B_try
            c1 = ComputeCost_BN(X, Y, paras_try, lamda)

            B_try = np.array(paras["B"][i])
            B_try[j] += h
            paras_try["B"][i] = B_try
            c2 = ComputeCost_BN(X, Y, paras_try, lamda)

            grad_B[j] = (c2 - c1) / (2 * h)

        grad_B_list.append(grad_B)

    update_para = {'grad_W': grad_W_list, 'grad_b': grad_b_list, 'grad_G': grad_G_list, 'grad_B': grad_B_list}

    return update_para

def ComputeGradients(X, Y, paras, lamda):
    grad_W_list = []
    grad_b_list = []
    X_list = []

    X_list.append(X)
    for i in range(len(paras["W"]) - 1):
        S1 = np.dot(paras["W"][i], X) + paras["b"][i]
        X = np.maximum(0, S1)
        X_list.append(X)

    S = np.dot(paras["W"][-1], X) + paras["b"][-1]
    P = softmax(S)
    G = -(Y - P)

    for j in range(len(paras["W"]) - 1):
        grad_W = np.dot(G, X_list[-(j + 1)].T) / G.shape[1] + 2 * lamda * paras["W"][-(j + 1)]
        grad_b = np.dot(G, np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]
        grad_W_list.append(grad_W)
        grad_b_list.append(grad_b)

        G = np.dot(paras["W"][-(j + 1)].T, G)
        h = X_list[-(j + 1)]
        H = np.where(h == 0, h, 1)
        G = np.multiply(G, H)

    grad_W = np.dot(G, X_list[0].T) / G.shape[1] + 2 * lamda * paras["W"][0]
    grad_b = np.dot(G, np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]
    grad_W_list.append(grad_W)
    grad_b_list.append(grad_b)

    update_para = {'grad_W': grad_W_list, 'grad_b': grad_b_list}
    return update_para

def BatchNormBackPass(G, S_list, S_mean, S_var):
    theta = 1e-10
    n = G.shape[1]
    I_n = np.ones(n).reshape(-1, 1)
    sigma1 = np.power(S_var + theta, -0.5).reshape(-1, 1)
    sigma2 = np.power(S_var + theta, -1.5).reshape(-1, 1)
    G1 = np.multiply(G, sigma1)
    G2 = np.multiply(G, sigma2)
    D = S_list - S_mean.reshape(-1, 1)
    c = np.dot(np.multiply(G2, D), I_n)
    G = G1 - np.dot(np.dot(G1, I_n), I_n.T) / n - np.multiply(D, np.dot(c, I_n.T)) / n
    return G

def ComputeGradients_BN(X, Y, paras, lamda, mean_av_list, var_av_list):
    grad_W_list = []
    grad_b_list = []
    grad_G_list = []
    grad_B_list = []
    X_list = []
    S_list = []
    S_mean_list = []
    S_var_list = []
    S_norm_list = []
    alpha = 0.85
    theta = 1e-10

    X_list.append(X)
    for i in range(len(paras["W"]) - 1):
        S1 = np.dot(paras["W"][i], X) + paras["b"][i]
        S_mean = np.mean(S1, axis = 1)
        S_var = np.var(S1, axis = 1)
        S_norm = (S1 - S_mean[:,None]) / np.power(S_var[:,None] + theta, 0.5)
        S_rescale = np.multiply(paras["G"][i], S_norm) + paras["B"][i]
        X = np.maximum(0, S_rescale)
        S_list.append(S1)
        S_mean_list.append(S_mean)
        S_var_list.append(S_var)
        S_norm_list.append(S_norm)
        X_list.append(X)

    S = np.dot(paras["W"][-1], X) + paras["b"][-1]
    P = softmax(S)
    G = -(Y - P)

    for j in range(len(paras["W"]) - 1):
        grad_W = np.dot(G, X_list[-(j + 1)].T) / G.shape[1] + 2 * lamda * paras["W"][-(j + 1)]
        grad_b = np.dot(G, np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]
        grad_W_list.append(grad_W)
        grad_b_list.append(grad_b)

        G = np.dot(paras["W"][-(j + 1)].T, G)
        h = X_list[-(j + 1)]
        H = np.where(h == 0, h, 1)
        G = np.multiply(G, H)

        grad_G = np.dot(np.multiply(G, S_norm_list[-(j + 1)]), np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]
        grad_B = np.dot(G, np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]
        grad_G_list.append(grad_G)
        grad_B_list.append(grad_B)

        G = np.multiply(G, paras["G"][-(j + 1)])
        G = BatchNormBackPass(G, S_list[-(j + 1)], S_mean_list[-(j + 1)], S_var_list[-(j + 1)])

    grad_W = np.dot(G, X_list[0].T) / G.shape[1] + 2 * lamda * paras["W"][0]
    grad_b = np.dot(G, np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]
    grad_W_list.append(grad_W)
    grad_b_list.append(grad_b)

    if len(mean_av_list) == 0:
        mean_av_list = S_mean_list
        var_av_list = S_var_list
    else:
        for index in range(len(mean_av_list)):
            mean_av_list[index] = alpha * mean_av_list[index] + (1 - alpha) * S_mean_list[index]
            var_av_list[index] = alpha * var_av_list[index] + (1 - alpha) * S_var_list[index]

    update_para = {'grad_W': grad_W_list, 'grad_b': grad_b_list, 'grad_G': grad_G_list, 'grad_B': grad_B_list}
    return update_para, mean_av_list, var_av_list

def MiniBatchGD_BN(X_train, Y_train, X_val, Y_val, paras, lamda, n_batch, eta_min, eta_max, n_s, n_epochs):
    y_train =  normal_representation(Y_train)
    y_val = normal_representation(Y_val)

    train_cost_list = []
    val_cost_list = []
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    mean_av_list = []
    var_av_list = []

    # train_cost = ComputeCost(X_train, Y_train, paras, lamda)
    # train_cost_list.append(train_cost)
    # val_cost = ComputeCost(X_val, Y_val, paras, lamda)
    # val_cost_list.append(val_cost)
    #
    # train_acc = ComputeAccuracy(X_train, y_train, paras)
    # train_acc_list.append(train_acc)
    # val_acc = ComputeAccuracy(X_val, y_val, paras)
    # val_acc_list.append(val_acc)

    for i in range(n_epochs):
        col_idx = np.random.permutation(X_train.shape[1])
        shuffled_X = X_train[:,col_idx]
        shuffled_Y = Y_train[:,col_idx]
        n = shuffled_X.shape[1]
        loop = n // n_batch
        for j in range(loop):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            Xbatch = shuffled_X[:, j_start:j_end]
            Ybatch = shuffled_Y[:, j_start:j_end]
            update_para, mean_av_list, var_av_list = ComputeGradients_BN(Xbatch, Ybatch, paras, lamda, mean_av_list, var_av_list)
            if i * loop < n_s:
                eta_t = eta_min + (i * loop + j) / n_s * (eta_max - eta_min)
            elif 1 * n_s <= i * loop < 2 * n_s:
                eta_t = eta_max - (i * loop + j - n_s) / n_s * (eta_max - eta_min)
            elif 2 * n_s <= i * loop < 3 * n_s:
                eta_t = eta_min + (i * loop + j - 2 * n_s) / n_s * (eta_max - eta_min)
            elif 3 * n_s <= i * loop < 4 * n_s:
                eta_t = eta_max - (i * loop + j - 3 * n_s) / n_s * (eta_max - eta_min)
            elif 4 * n_s <= i * loop < 5 * n_s:
                eta_t = eta_min + (i * loop + j - 4 * n_s) / n_s * (eta_max - eta_min)
            elif 5 * n_s <= i * loop < 6 * n_s:
                eta_t = eta_max - (i * loop + j - 5 * n_s) / n_s * (eta_max - eta_min)
            elif 6 * n_s <= i * loop < 7 * n_s:
                eta_t = eta_min + (i * loop + j - 6 * n_s) / n_s * (eta_max - eta_min)
            elif 7 * n_s <= i * loop < 8 * n_s:
                eta_t = eta_max - (i * loop + j - 7 * n_s) / n_s * (eta_max - eta_min)
            elif 8 * n_s <= i * loop < 9 * n_s:
                eta_t = eta_min + (i * loop + j - 8 * n_s) / n_s * (eta_max - eta_min)
            elif 9 * n_s <= i * loop < 10 * n_s:
                eta_t = eta_max - (i * loop + j - 9 * n_s) / n_s * (eta_max - eta_min)
            else:
                break;

            for k in range(len(paras["W"])):
                paras["W"][k] -= eta_t * update_para["grad_W"][-(k + 1)]
                paras["b"][k] -= eta_t * update_para["grad_b"][-(k + 1)]
            for h in range(len(paras["G"])):
                paras["G"][h] -= eta_t * update_para["grad_G"][-(h + 1)]
                paras["B"][h] -= eta_t * update_para["grad_B"][-(h + 1)]

        train_cost = ComputeCost_Test(X_train, Y_train, paras, lamda, mean_av_list, var_av_list)
        train_cost_list.append(train_cost)
        val_cost = ComputeCost_Test(X_val, Y_val, paras, lamda, mean_av_list, var_av_list)
        val_cost_list.append(val_cost)

        train_acc = ComputeAccuracy_Test(X_train, y_train, paras, mean_av_list, var_av_list)
        train_acc_list.append(train_acc)
        val_acc = ComputeAccuracy_Test(X_val, y_val, paras, mean_av_list, var_av_list)
        val_acc_list.append(val_acc)

        print(len(train_cost_list))
    plt.figure(figsize = (15.0, 5.0))
    plt.subplot(121)
    plt.plot(train_cost_list, 'g', label = "training cost")
    plt.plot(val_cost_list, 'r', label = "validation cost")
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.xlim(xmin = 0)
    plt.text(5.0, train_cost_list[-1] + 0.8, "Final training cost: " + str(round(train_cost_list[-1], 3)))
    plt.text(5.0, train_cost_list[-1] + 0.7, "Final validation cost: " + str(round(val_cost_list[-1], 3)))
    plt.title('Traning and Validation Cost with Lamda: ' + str(round(lamda, 6)))
    plt.legend(loc = "best")

    plt.subplot(122)
    plt.plot(train_acc_list, 'g', label = "training accuracy")
    plt.plot(val_acc_list, 'r', label = "validation accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim(xmin = 0)
    plt.text(5.0, train_acc_list[-1] - 0.1, "Final training accuracy: " + str(round(train_acc_list[-1], 3)))
    plt.text(5.0, train_acc_list[-1] - 0.15, "Final validation accuracy: " + str(round(val_acc_list[-1], 3)))
    plt.title('Traning and Validation Accuracy with Lamda: ' + str(round(lamda, 6)))
    plt.legend(loc = "best")

    plt.savefig(str(round(lamda, 6)) + "image.png")
    plt.show()
    return paras, mean_av_list, var_av_list

origin_dataset = LoadBatch('Datasets/cifar-10-batches-py/data_batch_1')
total_raw_images = np.transpose(origin_dataset[bytes("data", "utf-8")] / 255.0)
total_labels = np.array(origin_dataset[bytes("labels", "utf-8")])
for i in range(4):
    dataset = LoadBatch('Datasets/cifar-10-batches-py/data_batch_' + str(i + 2))
    raw_images = np.transpose(dataset[bytes("data", "utf-8")] / 255.0)
    labels = dataset[bytes("labels", "utf-8")]
    total_raw_images = np.concatenate((total_raw_images, raw_images), axis = 1)
    total_labels = np.concatenate((total_labels, labels), axis=None)
test_dataset = LoadBatch('Datasets/cifar-10-batches-py/test_batch')

train_raw_images = total_raw_images[:, 0: total_raw_images.shape[1] - 5000]
val_raw_images = total_raw_images[:, total_raw_images.shape[1] - 5000: total_raw_images.shape[1]]
test_raw_images = np.transpose(test_dataset[bytes("data", "utf-8")] / 255.0)

train_labels = total_labels[0: len(total_labels) - 5000]
val_labels = total_labels[len(total_labels) - 5000: len(total_labels)]
test_labels = test_dataset[bytes("labels", "utf-8")]

train_one_hot_labels = one_hot_representation(train_labels)
val_one_hot_labels = one_hot_representation(val_labels)
test_one_hot_labels = one_hot_representation(test_labels)

train_norm_imgs = Normalization(train_raw_images)
val_norm_imgs = Normalization(val_raw_images)
test_norm_imgs = Normalization(test_raw_images)

paras = Initialization(dims, True)
lamda = 0.005

final_para, mean_av_list, var_av_list = MiniBatchGD_BN(train_norm_imgs, train_one_hot_labels, val_norm_imgs, val_one_hot_labels, paras, lamda, 100, 1e-5, 1e-1, 2250, 20)
acc = ComputeAccuracy_Test(test_norm_imgs, test_labels, final_para, mean_av_list, var_av_list)
print(acc)

#update_para1 = ComputeGradsNum(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], paras, 0, 1e-5)
#update_para2 = ComputeGradsNumSlow(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], paras, 0, 1e-5)
#update_para3 = ComputeGradients(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], paras, 0)
# update_para4 = ComputeGradsNum_BN(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], paras, 0, 1e-5)
# update_para5, mean_av, std_av = ComputeGradients_BN(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], paras, 0, [], [])
# diff_W, diff_b, diff_G, diff_B = 0, 0, 0, 0
# for i in range(len(update_para4["grad_W"])):
#     diff_W += np.sum(np.absolute(update_para5["grad_W"][i] - update_para4["grad_W"][-(i + 1)]))
#     diff_b += np.sum(np.absolute(update_para5["grad_b"][i] - update_para4["grad_b"][-(i + 1)]))
# for i in range(len(update_para4["grad_G"])):
#     diff_G += np.sum(np.absolute(update_para5["grad_G"][i] - update_para4["grad_G"][-(i + 1)]))
#     diff_B += np.sum(np.absolute(update_para5["grad_B"][i] - update_para4["grad_B"][-(i + 1)]))
# print([diff_W, diff_b, diff_G, diff_B])
# print([update_para5["grad_W"][5] - update_para4["grad_W"][-6], update_para5["grad_b"][5] - update_para4["grad_b"][-6]])
# print([update_para5["grad_G"][1] - update_para4["grad_G"][-2], update_para5["grad_B"][1] - update_para4["grad_B"][-2]])
#print([update_para1["grad_W"][0] - update_para2["grad_W"][0], update_para1["grad_b"][0] - update_para2["grad_b"][0], update_para1["grad_W"][1] - update_para2["grad_W"][1], update_para1["grad_b"][1] - update_para2["grad_b"][1]])
#print([update_para3["grad_W"][1] - update_para1["grad_W"][0], update_para3["grad_b"][1] - update_para1["grad_b"][0], update_para3["grad_W"][0] - update_para1["grad_W"][1], update_para3["grad_b"][0] - update_para1["grad_b"][1]])
#print([update_para5["grad_W"][-1] - update_para4["grad_W"][0], update_para5["grad_b"][-1] - update_para4["grad_b"][0], update_para5["grad_W"][-2] - update_para4["grad_W"][1], update_para5["grad_b"][-2] - update_para4["grad_b"][1]])
#print([update_para3["grad_W"][1] - update_para1["grad_W"][0], update_para3["grad_b"][1] - update_para1["grad_b"][0], update_para3["grad_W"][0] - update_para1["grad_W"][1], update_para3["grad_b"][0] - update_para1["grad_b"][1]])
# aa = np.array([[1, 2, 3]])
# bb = np.array([[4, 5, 6]])
# cc = [aa, bb]
# print(cc[1])
# print(len(cc))
aa = np.array([[1, 2, 3], [4, 5, 6]])
cc = np.array([2, 3]).reshape(-1, 1).T
# bb = np.multiply(aa, cc)
print(aa - cc.T)
