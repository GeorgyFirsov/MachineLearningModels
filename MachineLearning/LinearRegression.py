from collections.abc import Iterable
from typing import Union

import numpy as np

from MachineLearning.Utils import (generate_polynomials,
                                   design_matrix,
                                   LinearRegressionParameters,
                                   )


class LinearRegression(object):
    """This class represents linear regression model.
    It constructs generalized linear regression with
    polynomials as functions of variable X:
        f(x) = b1 + b2 * x + ... + bn * x ^ (n-1)

    Public member functions:
        - fit - call this function to fit training data to model
        - predict - call this function to predict value of Y with trained model
        - parameters - call this function to obtain model's hyper-parameters
    """

    def __init__(self, *, degree: int = 1):
        """Initializes basic model parameters.
        Receives only degree of underlying regression polynomial.

        :param degree: degree of regression polynomial (only named parameter).
                       1 by default for linear regression function
        """

        # Check if degree is greater or equal to zero
        if degree <= 0:
            raise ValueError("Degree must be a positive integer")

        # Generate functions
        self.__functions = generate_polynomials(degree)

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

        # x-values and y-values must be represented as iterable objects
        if not isinstance(x_values, Iterable) or not isinstance(y_values, Iterable):
            raise TypeError("X and Y values should be iterables")

        self.__x_values = x_values
        self.__y_values = y_values

        # Convert values to numpy arrays if necessary
        if not isinstance(self.__x_values, np.ndarray):
            self.__x_values = np.array(self.__x_values)
        if not isinstance(self.__y_values, np.ndarray):
            self.__y_values = np.array(self.__y_values)

        # Check if X and Y have the same length
        if len(self.__x_values) != len(self.__y_values):
            raise ValueError("X and Y values' iterables must be the same length")

        # Check if this length is more than 1
        if len(self.__x_values) <= 1:
            raise ValueError("X and Y values' iterables must have at least 2 elements each")

        self.__train()

    def predict(self, x: Union[Iterable, float]) -> Union[Iterable, float]:
        """Predicts a value of regression function on specified value or iterable
        of values. If X is single value, output will be a single value, otherwise
        it'll be numpy array of values of regression function on each value in X.

        :param x: value or iterable of values to put into regression function
        :return: value or numpy array of values of regression function on X
        """
        if isinstance(x, float):
            return self.__calculate_value(x)
        else:
            return np.array([self.__calculate_value(xi) for xi in x])

    @property
    def parameters(self) -> LinearRegressionParameters:
        """Method that allows user to obtain linear model hyper-parameters.

        :return: LinearRegressionParameters instance
        """

        # Check if model is already trained
        if self.__coefficients is None:
            raise ValueError("Fit values first")

        return LinearRegressionParameters(
            tuple(self.__coefficients),
            tuple(self.__functions)
        )

    def __train(self) -> None:
        # Fit must be succeeded before invocation of this function
        if self.__x_values is None or \
           self.__y_values is None:
            raise ValueError("Fit values first")

        # Here y-values and x-values must be represented as numpy arrays
        self.__design_matrix = design_matrix(self.__functions, self.__x_values)

        # Calculation of coefficients of regression function by following formula:
        #       b = inv(transpose(F) · F) · transpose(F) · y,
        # where F is design matrix, y - vector of y-values, b - vector of desired coefficients
        self.__coefficients = np.linalg.inv(
            self.__design_matrix.T.dot(self.__design_matrix)
        ).dot(self.__design_matrix.T).dot(self.__y_values)

    def __calculate_value(self, x: float) -> float:
        # Check if model is already trained
        if self.__coefficients is None:
            raise ValueError("Fit values first")

        return sum([
            b * y(x) for b, y in zip(self.__coefficients, self.__functions)
        ])
