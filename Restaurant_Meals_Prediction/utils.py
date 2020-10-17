import numpy as np
from sklearn.linear_model import LinearRegression


def mean_absolute_percentage_error(y_true, y_pred):
    """
    This function creates/generates mean absolute percentage error metric
    :param y_true: true value of a target vector
    :param y_pred: predicted value of the target vector
    :return: metric value (int)
    """


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


def time_lags_generation(df, start, end):

    """
    This function generates a dataframe with each feature is a lags of time series of a target vector.

    Lags of time series - Shifting the series  𝑛  steps back, we get a feature column where the current value of time
    series is aligned with its value at time  𝑡−𝑛 . If we make a 1 lag shift and train a model on that feature,
    the model will be able to forecast 1 step ahead from having observed the current state of the series.
    Increasing the lag, say, up to 6, will allow the model to make predictions 6 steps ahead; however it will use data
    observed 6 steps back. If something fundamentally changes the series during that unobserved period, the model will
    not catch these changes and will return forecasts with a large error. Therefore, during the initial lag selection,
    one has to find a balance between the optimal prediction quality and the length of the forecasting horizon.



    :param df: one dimensional dataframe (time series)
    :param y: column to create features from
    :param start: starting value for time shift
    :param end: final value for time shift
    :return: data frame with shifted series
    """

    for i in range(start, end):
        df["lag_{}".format(i)] = df.target.shift(i)

    return df


def compute_lr_error(x_train, y_train, x_test, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    prediction = lr.predict(x_test)
    error = mean_absolute_percentage_error(prediction, y_test)

    return error

