import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def timeseries_train_test_split(x, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """

    test_index = int(len(x) * (1 - test_size))

    x_train = x.iloc[:test_index]
    y_train = y.iloc[:test_index]
    x_test = x.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return x_train, x_test, y_train, y_test


def train_test_split(x):
    target = x.dropna().target
    x = x.dropna().drop(['target'], axis=1)
    x_train, x_test, y_train, y_test = timeseries_train_test_split(x, target, test_size=0.3)

    return x_train, x_test, y_train, y_test
