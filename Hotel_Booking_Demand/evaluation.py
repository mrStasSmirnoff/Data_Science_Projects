from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def cv_roc_auc_acc(model, x_train, y_train, scoring, cv):
    cross_val = cross_validate(model, x_train, y=y_train, scoring=scoring, cv=cv, n_jobs=-1)

    print('ROC AUC value : %f' % (cross_val['test_auc'].mean()))
    print('Accuracy value : %f' % (cross_val['test_acc'].mean()))


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
