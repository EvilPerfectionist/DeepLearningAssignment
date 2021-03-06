import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import tensorflow as tf

dim = 3072
num_labs = 10
dims = [3072, 50, 50, 10]

def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

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
    alpha = 0.9
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
            X_data = Xbatch.T.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
            X_dataset = tf.data.Dataset.from_tensor_slices(X_data)
            augmentations = [color, zoom]
            for f in augmentations:
                X_dataset = X_dataset.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: f(x), lambda: x), num_parallel_calls=8)
            X_dataset = X_dataset.map(lambda x: tf.clip_by_value(x, 0, 1))
            new_Xbatch = np.empty((3072, 0))
            for z in X_dataset:
                back = z.numpy().transpose(2, 0, 1).reshape(3072, 1)
                new_Xbatch = np.append(new_Xbatch, back, axis = 1)
            update_para, mean_av_list, var_av_list = ComputeGradients_BN(new_Xbatch, Ybatch, paras, lamda, mean_av_list, var_av_list)
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
    #plt.show()
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
lamda = 10 ** -2.18
final_para, mean_av_list, var_av_list = MiniBatchGD_BN(train_norm_imgs, train_one_hot_labels, val_norm_imgs, val_one_hot_labels, paras, lamda, 100, 1e-5, 1e-1, 2250, 20)
acc = ComputeAccuracy_Test(test_norm_imgs, test_labels, final_para, mean_av_list, var_av_list)
print(acc)
