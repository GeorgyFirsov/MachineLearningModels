import numpy as np


def mean_squared_error(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculates MSE score on predicted by model and test samples.

    :param y_pred: sample of predicted values
    :param y_test: sample of observed test values
    :return: MSE score
    """

    # Check if y_pred and y_test have equal length
    if len(y_pred) != len(y_test):
        raise ValueError("Predicted and test values must have equal shape")

    return (sum((y_p - y_t) ** 2 for y_p, y_t in zip(y_pred, y_test)) / len(y_pred))[0]


def root_mean_squared_error(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculates RMSE score on predicted by model and test samples.

    :param y_pred: sample of predicted values
    :param y_test: sample of observed test values
    :return: RMSE score
    """
    return np.sqrt(mean_squared_error(y_pred, y_test))


def mean_absolute_error(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculates MAE score on predicted by model and test samples.

    :param y_pred: sample of predicted values
    :param y_test: sample of observed test values
    :return: MAE score
    """

    # Check if y_pred and y_test have equal length
    if len(y_pred) != len(y_test):
        raise ValueError("Predicted and test values must have equal shape")

    return (sum(np.abs(y_p - y_t) for y_p, y_t in zip(y_pred, y_test)) / len(y_pred))[0]
