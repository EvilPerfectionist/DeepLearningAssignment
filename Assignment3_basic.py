import numpy as np
import matplotlib.pyplot as plt
import math

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

def ComputeGradsNum(X, Y, W1, b1, W2, b2, lamda, h):
	""" Converted from matlab code """

	grad_W1 = np.zeros(W1.shape);
	grad_b1 = np.zeros((len(b1), 1));
	grad_W2 = np.zeros(W2.shape);
	grad_b2 = np.zeros((len(b2), 1));

	c = ComputeCost(X, Y, W1, b1, W2, b2, lamda);

	for i in range(len(b1)):
		b1_try = np.array(b1)
		b1_try[i] += h
		c2 = ComputeCost(X, Y, W1, b1_try, W2, b2, lamda)
		grad_b1[i] = (c2 - c) / h

	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W1_try = np.array(W1)
			W1_try[i,j] += h
			c2 = ComputeCost(X, Y, W1_try, b1, W2, b2, lamda)
			grad_W1[i,j] = (c2-c) / h

	for i in range(len(b2)):
		b2_try = np.array(b2)
		b2_try[i] += h
		c2 = ComputeCost(X, Y, W1, b1, W2, b2_try, lamda)
		grad_b2[i] = (c2 - c) / h

	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W2_try = np.array(W2)
			W2_try[i,j] += h
			c2 = ComputeCost(X, Y, W1, b1, W2_try, b2, lamda)
			grad_W2[i,j] = (c2-c) / h

	return [grad_W1, grad_b1, grad_W2, grad_b2]

def ComputeGradsNumSlow(X, Y, W1, b1, W2, b2, lamda, h):
	""" Converted from matlab code """

	grad_W1 = np.zeros(W1.shape);
	grad_b1 = np.zeros((len(b1), 1));
	grad_W2 = np.zeros(W2.shape);
	grad_b2 = np.zeros((len(b2), 1));

	c = ComputeCost(X, Y, W1, b1, W2, b2, lamda);

	for i in range(len(b1)):
		b1_try = np.array(b1)
		b1_try[i] -= h
		c1 = ComputeCost(X, Y, W1, b1_try, W2, b2, lamda)

		b1_try = np.array(b1)
		b1_try[i] += h
		c2 = ComputeCost(X, Y, W1, b1_try, W2, b2, lamda)

		grad_b1[i] = (c2 - c1) / (2 * h)

	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W1_try = np.array(W1)
			W1_try[i,j] -= h
			c1 = ComputeCost(X, Y, W1_try, b1, W2, b2, lamda)

			W1_try = np.array(W1)
			W1_try[i,j] += h
			c2 = ComputeCost(X, Y, W1_try, b1, W2, b2, lamda)

			grad_W1[i,j] = (c2 - c1) / (2 * h)

	for i in range(len(b2)):
		b2_try = np.array(b2)
		b2_try[i] -= h
		c1 = ComputeCost(X, Y, W1, b1, W2, b2_try, lamda)

		b2_try = np.array(b2)
		b2_try[i] += h
		c2 = ComputeCost(X, Y, W1, b1, W2, b2_try, lamda)

		grad_b2[i] = (c2 - c1) / (2 * h)

	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W2_try = np.array(W2)
			W2_try[i,j] -= h
			c1 = ComputeCost(X, Y, W1, b1, W2_try, b2, lamda)

			W2_try = np.array(W2)
			W2_try[i,j] += h
			c2 = ComputeCost(X, Y, W1, b1, W2_try, b2, lamda)

			grad_W2[i,j] = (c2 - c1) / (2 * h)

	return [grad_W1, grad_b1, grad_W2, grad_b2]

def ComputeGradients(X, Y, paras, lamda):
    s1 = np.dot(paras["W"][0], X) + paras["b"][0]
    h = np.maximum(0, s1)
    s = np.dot(paras["W"][1], h) + paras["b"][1]
    P = softmax(s)
    G = -(Y - P)
    grad_W2 = np.dot(G, h.T) / G.shape[1] + 2 * lamda * paras["W"][1]
    grad_b2 = np.dot(G, np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]

    G = np.dot(paras["W"][1].T, G)
    H = np.where(h == 0, h, 1)
    G = np.multiply(G, H)
    grad_W1 = np.dot(G, X.T) / G.shape[1] + 2 * lamda * paras["W"][0]
    grad_b1 = np.dot(G, np.ones(G.shape[1]).reshape(-1, 1)) / G.shape[1]
    return grad_W1, grad_b1, grad_W2, grad_b2

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
            grad_W1, grad_b1, grad_W2, grad_b2 = ComputeGradients(Xbatch, Ybatch, paras, lamda)
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

            paras["W"][0] -= eta_t * grad_W1
            paras["b"][0] -= eta_t * grad_b1
            paras["W"][1] -= eta_t * grad_W2
            paras["b"][1] -= eta_t * grad_b2

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
    return W1, b1, W2, b2

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

final_W1, final_b1, final_W2, final_b2 = MiniBatchGD(train_norm_imgs, train_one_hot_labels, val_norm_imgs, val_one_hot_labels, paras, lamda, 100, 1e-5, 1e-1, 980, 12)
# acc = ComputeAccuracy(test_norm_imgs, test_labels, final_W1, final_b1, final_W2, final_b2)
# print(acc)

# grad_list1 = ComputeGradsNum(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], training_paras, 0, 1e-5)
# grad_list2 = ComputeGradsNumSlow(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], training_paras, 0, 1e-5)
# grad_list3 = ComputeGradients(train_norm_imgs[:, 1:10], train_one_hot_labels[:, 1:10], training_paras, 0)
# print([grad_list3[0] - grad_list2[0], grad_list3[1] - grad_list2[1], grad_list3[2] - grad_list2[2], grad_list3[3] - grad_list2[3]])

# aa = np.array([[1, 2, 3]])
# bb = np.array([[4, 5, 6]])
# cc = [aa, bb]
# print(cc[1])
# print(len(cc))
