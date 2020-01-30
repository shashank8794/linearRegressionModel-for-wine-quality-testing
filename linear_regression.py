import numpy as np

def load_data(path, num_train):
    wine = np.loadtxt('winequality-white.csv',dtype=float,skiprows=1)
    X_train = wine[:num_train, :11]
    Y_train = wine[:num_train,11:]
    X_test = wine[num_train:,:11]
    Y_test = wine[num_train:,11:]
    X_train_dummy = np.ones((num_train,12))
    X_train_dummy[:num_train,:11] = X_train
    X_test_dummy = np.ones((X_test.shape[0],12))
    X_test_dummy[:X_test.shape[0],:11] = X_test
    
    return X_train, Y_train, X_test, Y_test


def fit(X, Y):
    X_train = X.transpose()
    Y_train = Y.transpose()
    a = np.linalg.inv(X_train.dot(X)).dot((Y.dot(X)).transpose())
    
    return a


def predict(X, theta):
    b = X.dot(theta)
    
    return b


def energy(Y_pred, Y_gt):
    c = np.sum(np.square(Y_pred - Y_gt))

    return c