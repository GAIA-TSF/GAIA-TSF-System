import numpy as np

"""Evaluation metrics for regression tasks. 
"""


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))


def mape(y, y_pred):
    return np.mean(np.abs((y - y_pred) / y)) * 100
