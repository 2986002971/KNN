from numba import jit, prange
import numpy as np
import time


@jit(nopython=True, parallel=True)
def knn_predict(x_train, y_train, x_test, k):
    x_train = np.ascontiguousarray(x_train)
    y_train = np.ascontiguousarray(y_train)
    x_test = np.ascontiguousarray(x_test)

    n_samples_train = x_train.shape[0]
    n_samples_test = x_test.shape[0]
    y_predict = np.zeros(n_samples_test)

    for i in prange(n_samples_test):
        distances = np.sum((x_train - x_test[i]) ** 2, axis=1)
        nearest = np.argsort(distances)[:k]
        y_predict[i] = np.argmax(np.bincount(y_train[nearest]))

    return y_predict


@jit(nopython=True, parallel=True)
def knn_predict_manhattan(x_train, y_train, x_test, k):
    x_train = np.ascontiguousarray(x_train)
    y_train = np.ascontiguousarray(y_train)
    x_test = np.ascontiguousarray(x_test)

    n_samples_train = x_train.shape[0]
    n_samples_test = x_test.shape[0]
    y_predict = np.zeros(n_samples_test)

    for i in prange(n_samples_test):
        distances = np.sum(np.abs(x_train - x_test[i]), axis=1)
        nearest = np.argsort(distances)[:k]
        y_predict[i] = np.argmax(np.bincount(y_train[nearest]))

    return y_predict


def knn_predict_chebyshev(x_train, y_train, x_test, k):
    x_train = np.ascontiguousarray(x_train)
    y_train = np.ascontiguousarray(y_train)
    x_test = np.ascontiguousarray(x_test)

    n_samples_train = x_train.shape[0]
    n_samples_test = x_test.shape[0]
    y_predict = np.zeros(n_samples_test)

    for i in prange(n_samples_test):
        distances = np.max(np.abs(x_train - x_test[i]), axis=1)
        nearest = np.argsort(distances)[:k]
        y_predict[i] = np.argmax(np.bincount(y_train[nearest]))

    return y_predict


@jit(nopython=True, parallel=True)
def generate_data(n_samples):
    X = np.random.rand(n_samples, 2) * 10
    Y = np.zeros(n_samples, dtype=np.int64)
    for i in prange(n_samples):
        if 0 < X[i, 0] < 3 and 0 < X[i, 1] < 3:
            Y[i] = 1
        elif 0 < X[i, 0] < 3 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 2
        elif 0 < X[i, 0] < 3 and 7 < X[i, 1] < 10:
            Y[i] = 3
        elif 3.5 < X[i, 0] < 6.5 and 0 < X[i, 1] < 3:
            Y[i] = 4
        elif 3.5 < X[i, 0] < 6.5 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 5
        elif 3.5 < X[i, 0] < 6.5 and 7 < X[i, 1] < 10:
            Y[i] = 6
        elif 7 < X[i, 0] < 10 and 0 < X[i, 1] < 3:
            Y[i] = 7
        elif 7 < X[i, 0] < 10 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 8
        elif 7 < X[i, 0] < 10 and 7 < X[i, 1] < 10:
            Y[i] = 9
    valid = Y > 0
    X = X[valid]
    Y = Y[valid]
    return X, Y


def add_noise(X, Y, n_noise):
    """
    向数据集添加噪声点。

    Parameters:
    - X: 原始数据的特征数组。
    - Y: 原始数据的标签数组。
    - n_noise: 要添加的噪声点数量。
    - n_classes: 类别总数，默认为9。

    Returns:
    - X_noise: 包含噪声点的新特征数组。
    - Y_noise: 包含噪声点的新标签数组。
    """
    X_noise = np.random.rand(n_noise, 2) * 10
    Y_noise = np.random.randint(0, 9, size=n_noise)

    # 将噪声点合并到原始数据集中
    X_combined = np.vstack((X, X_noise))
    Y_combined = np.concatenate((Y, Y_noise))

    return X_combined, Y_combined
