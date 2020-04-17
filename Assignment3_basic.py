import numpy as np
import matplotlib.pyplot as plt
import math
import copy

dim = 3072
num_labs = 10
dims = [3072, 50, 10]

def Initialization(dims):
    W_list = []
    b_list = []

    for i in range(len(dims) - 1):
        mu, sigma = 0, 1 / math.sqrt(dims[i])
        W = np.random.normal(mu, sigma, (dims[i + 1], dims[i]))
        b = np.zeros((dims[i + 1], 1))
        W_list.append(W)
        b_list.append(b)

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

def Normalization(raw_images):
    mean = np.mean(raw_images, axis = 1)
    std = np.std(raw_images, axis = 1)
    norm = (raw_images - mean[:,None]) / std[:,None]
    return norm

def EvaluateClassifier(X, paras):
    s1 = np.dot(paras["W"][0], X) + paras["b"][0]
    h = np.maximum(0, s1)
    s = np.dot(paras["W"][1], h) + paras["b"][1]
    p = softmax(s)
    return p

def ComputeAccuracy(X, y, paras):
    p = EvaluateClassifier(X, paras)
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

def MiniBatchGD(X_train, Y_train, X_val, Y_val, paras, lamda, n_batch, eta_min, eta_max, n_s, n_epochs):
    y_train =  normal_representation(Y_train)
    y_val = normal_representation(Y_val)

    train_cost_list = []
    val_cost_list = []
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    train_cost = ComputeCost(X_train, Y_train, paras, lamda)
    train_cost_list.append(train_cost)
    val_cost = ComputeCost(X_val, Y_val, paras, lamda)
    val_cost_list.append(val_cost)

    train_loss = ComputeLoss(X_train, Y_train, paras)
    train_loss_list.append(train_loss)
    val_loss = ComputeLoss(X_val, Y_val, paras)
    val_loss_list.append(val_loss)

    train_acc = ComputeAccuracy(X_train, y_train, paras)
    train_acc_list.append(train_acc)
    val_acc = ComputeAccuracy(X_val, y_val, paras)
    val_acc_list.append(val_acc)

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
            update_para = ComputeGradients(Xbatch, Ybatch, paras, lamda)
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

            if i in range(len(paras["W"])):
                paras["W"][i] = eta_t * update_para["grad_W"][-(i + 1)]
                paras["b"][i] = eta_t * update_para["grad_b"][-(i + 1)]

        train_cost = ComputeCost(X_train, Y_train, paras, lamda)
        train_cost_list.append(train_cost)
        val_cost = ComputeCost(X_val, Y_val, paras, lamda)
        val_cost_list.append(val_cost)

        train_loss = ComputeLoss(X_train, Y_train, paras)
        train_loss_list.append(train_loss)
        val_loss = ComputeLoss(X_val, Y_val, paras)
        val_loss_list.append(val_loss)

        train_acc = ComputeAccuracy(X_train, y_train, paras)
        train_acc_list.append(train_acc)
        val_acc = ComputeAccuracy(X_val, y_val, paras)
        val_acc_list.append(val_acc)

        print(len(train_cost_list))
    plt.figure(figsize = (20.0, 5.0))
    plt.subplot(131)
    plt.plot(train_cost_list, 'g', label = "training cost")
    plt.plot(val_cost_list, 'r', label = "validation cost")
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.xlim(xmin = 0)
    plt.text(5.0, train_cost_list[-1] + 0.8, "Final training cost: " + str(round(train_cost_list[-1], 3)))
    plt.text(5.0, train_cost_list[-1] + 0.7, "Final validation cost: " + str(round(val_cost_list[-1], 3)))
    plt.title('Traning and Validation Cost with Lamda: ' + str(round(lamda, 6)))
    plt.legend(loc = "best")

    plt.subplot(132)
    plt.plot(train_loss_list, 'g', label = "training loss")
    plt.plot(val_loss_list, 'r', label = "validation loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(xmin = 0)
    plt.text(5.0, train_loss_list[-1] + 0.8, "Final training loss: " + str(round(train_loss_list[-1], 3)))
    plt.text(5.0, train_loss_list[-1] + 0.7, "Final validation loss: " + str(round(val_loss_list[-1], 3)))
    plt.title('Traning and Validation Loss with Lamda: ' + str(round(lamda, 6)))
    plt.legend(loc = "best")

    plt.subplot(133)
    plt.plot(train_acc_list, 'g', label = "training accuracy")
    plt.plot(val_acc_list, 'r', label = "validation accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim(xmin = 0)
    plt.text(5.0, train_acc_list[-1] - 0.3, "Final training accuracy: " + str(round(train_acc_list[-1], 3)))
    plt.text(5.0, train_acc_list[-1] - 0.35, "Final validation accuracy: " + str(round(val_acc_list[-1], 3)))
    plt.title('Traning and Validation Accuracy with Lamda: ' + str(round(lamda, 6)))
    plt.legend(loc = "best")

    plt.savefig(str(round(lamda, 6)) + "image.png")
    plt.show()
    return paras

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

train_raw_images = total_raw_images[:, 0: total_raw_images.shape[1] - 1000]
val_raw_images = total_raw_images[:, total_raw_images.shape[1] - 1000: total_raw_images.shape[1]]
test_raw_images = np.transpose(test_dataset[bytes("data", "utf-8")] / 255.0)

train_labels = total_labels[0: len(total_labels) - 1000]
val_labels = total_labels[len(total_labels) - 1000: len(total_labels)]
test_labels = test_dataset[bytes("labels", "utf-8")]

train_one_hot_labels = one_hot_representation(train_labels)
val_one_hot_labels = one_hot_representation(val_labels)
test_one_hot_labels = one_hot_representation(test_labels)

train_norm_imgs = Normalization(train_raw_images)
val_norm_imgs = Normalization(val_raw_images)
test_norm_imgs = Normalization(test_raw_images)

paras = Initialization(dims)

lamda = 3.16e-4

final_para = MiniBatchGD(train_norm_imgs, train_one_hot_labels, val_norm_imgs, val_one_hot_labels, paras, lamda, 100, 1e-5, 1e-1, 980, 12)
# acc = ComputeAccuracy(test_norm_imgs, test_labels, final_W1, final_b1, final_W2, final_b2)
# print(acc)

#update_para1 = ComputeGradsNum(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], paras, 0, 1e-5)
#update_para2 = ComputeGradsNumSlow(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], paras, 0, 1e-5)
#update_para3 = ComputeGradients(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], paras, 0)
#print([update_para1["grad_W"][0] - update_para2["grad_W"][0], update_para1["grad_b"][0] - update_para2["grad_b"][0], update_para1["grad_W"][1] - update_para2["grad_W"][1], update_para1["grad_b"][1] - update_para2["grad_b"][1]])
#print([update_para3["grad_W"][1] - update_para1["grad_W"][0], update_para3["grad_b"][1] - update_para1["grad_b"][0], update_para3["grad_W"][0] - update_para1["grad_W"][1], update_para3["grad_b"][0] - update_para1["grad_b"][1]])
# aa = np.array([[1, 2, 3]])
# bb = np.array([[4, 5, 6]])
# cc = [aa, bb]
# print(cc[1])
# print(len(cc))

aa = [1, 2, 3]
print(aa[-3])
