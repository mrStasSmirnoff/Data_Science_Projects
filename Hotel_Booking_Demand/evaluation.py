from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def cv_roc_auc_acc(model, x_train, y_train, scoring, cv):
    """
    Evaluate metric(s) by cross-validation and also record fit/score times.

    :param model: training model
    :param x_train: The data to fit. Can be for example a list, or an array.
    :param y_train: The target variable to try to predict
    :param scoring: scoring function : 'r2', 'neg_mean_squared_error', 'accuracy' etc
    :param cv: cross-validation splitting strategy
    :return:
    """

    cross_val = cross_validate(model, x_train, y=y_train, scoring=scoring, cv=cv, n_jobs=-1)

    print('ROC AUC value : %f' % (cross_val['test_auc'].mean()))
    print('Accuracy value : %f' % (cross_val['test_acc'].mean()))

    return cross_val


def model_evaluation_classification(model, x_train, y_train, X_holdout, y_holdout):
    model.fit(x_train, y_train)
    prediction = model.predict(X_holdout)

    print('Accuracy : %f' % (accuracy_score(y_holdout, prediction)))
    print('ROC AUC : %f' % (roc_auc_score(y_holdout, prediction)))

    return model, prediction


def cv_rmse_mae(model, x_train, y_train, n_folds, random_state):
    kf = KFold(n_folds, shuffle=True, random_state=random_state)
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=kf))
    mae = -cross_val_score(model, x_train, y_train, scoring="neg_mean_absolute_error", cv=kf)

    return rmse, mae


def rmse(y_pred, y_true):

    return np.sqrt(mean_squared_error((y_true, y_pred)))


def model_evaluation_regression(model, x_train, y_train, x_holdout, y_holdout):
    model.fit(x_train, y_train)
    prediction = model.predict(x_holdout)

    print('MAE : %f' % (mean_absolute_error(y_holdout, prediction)))
    print('R2 : %f' % (r2_score(y_holdout, prediction)))
    print('RMSE : %f' % (rmse(y_holdout, prediction)))

    return model, prediction
