from collections.abc import Iterable
from typing import Optional, Union
import warnings

import numpy as np

from MachineLearning.Utils import (design_matrix_multiple,
                                   MultipleLinearRegressionParameters,
                                   )


class MultipleLinearRegression(object):
    """This class represents multiple linear regression model.
    It constructs multiple linear regression with
    identity as functions of variables xi:
        f(x) = b0 + b1 * x1 + ... + bn * xn

    Public member functions:
        - fit - call this function to fit training data to model
        - predict - call this function to predict value of Y with trained model
        - parameters - call this function to obtain model's hyper-parameters
    """

    def __init__(self, *, functions: Optional[np.ndarray] = None):
        """Initializes basic model parameters.
        Receives only degree of underlying regression polynomial.

        :param functions: functions of all features
        """

        # Non-linear on xi functions are not implemented for now and will be ignored
        if functions is not None:
            warnings.warn("Non-linear functions of variables are currently not implemented", RuntimeWarning)

        self.__functions = functions

        # These values will be filled later
        self.__design_matrix = None
        self.__x_values = None
        self.__y_values = None
        self.__coefficients = None

    def fit(self, x_values: Iterable, y_values: Iterable) -> None:
        """This functions receives x-values and y-values as iterables and
        invokes model training.

        :param x_values: iterable of x-values
        :param y_values: iterable of y-values
        """

        self.__x_values = x_values
        self.__y_values = y_values

        # Convert values to numpy arrays if necessary
        if not isinstance(self.__x_values, np.ndarray):
            self.__x_values = np.array(self.__x_values)
        if not isinstance(self.__y_values, np.ndarray):
            self.__y_values = np.array(self.__y_values)

        self.__train()

    def predict(self, x: Iterable) -> Union[np.ndarray, float]:
        """Predicts a value of regression function on specified value or iterable
        of values. If X is single vector, output will be a single value, otherwise
        it'll be numpy array of values of regression function on each value in X.

        :param x: value or iterable of values to put into regression function
        :return: value or numpy array of values of regression function on X
        """

        # Copy x to avoid its corruption
        internal_x = x

        # x-values must be in numpy array for further operations
        if not isinstance(internal_x, np.ndarray):
            internal_x = np.array(internal_x)

        # Return value depends on passed x.
        # It can be single vector or several vectors.
        try:
            return np.array([self.__calculate_value(xi) for xi in internal_x.T])
        except TypeError:
            return self.__calculate_value(internal_x)

    @property
    def parameters(self) -> MultipleLinearRegressionParameters:
        """Method that allows user to obtain linear model hyper-parameters.

        :return: MultipleLinearRegressionParameters instance
        """

        # Check if model is already trained
        if self.__coefficients is None:
            raise ValueError("Fit values first")

        return MultipleLinearRegressionParameters(tuple(self.__coefficients), tuple([self.__functions]))

    def __train(self) -> None:
        # Fit must be succeeded before invocation of this function
        if self.__x_values is None or \
           self.__y_values is None:
            raise ValueError("Fit values first")

        # Here y-values and x-values must be represented as numpy arrays
        self.__design_matrix = design_matrix_multiple(self.__x_values)

        # Calculation of coefficients of regression function by following formula:
        #       b = inv(transpose(F) · F) · transpose(F) · y,
        # where F is design matrix, y - vector of y-values, b - vector of desired coefficients
        self.__coefficients = np.linalg.inv(
            self.__design_matrix.T.dot(self.__design_matrix)
        ).dot(self.__design_matrix.T).dot(self.__y_values)

    def __calculate_value(self, x: np.ndarray) -> float:
        # Check if model is already trained
        if self.__coefficients is None:
            raise ValueError("Fit values first")

        return sum([b * xi for b, xi in zip(self.__coefficients[1:], x)]) + self.__coefficients[0]
