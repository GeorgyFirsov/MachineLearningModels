import numpy as np


def generate_polynomials(max_degree: int) -> np.ndarray:
    """Generates a certain amount of lambda-functions
    that represent polynomials with degree from 0 to max_degree.
    These polynomials are of following kind: x ^ i, where i is
    degree of the polynomial.

    :param max_degree: maximum degree of polynomial
    :return: numpy array of lambda-functions
    """

    # Check if degree is greater than zero
    if max_degree <= 0:
        raise ValueError("Cannot generate polynomial with non-positive degree")

    # Check if degree is an integer
    if not isinstance(max_degree, int):
        raise TypeError("Degree must be an integer")

    return np.array([
        lambda x, pwr=i: x ** pwr for i in range(max_degree + 1)
    ])


def design_matrix(functions: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """Calculates design matrix for linear regression by certain
    functions and x-values.

    :param functions: numpy array of lambda-functions
    :param x_values: values of variable X
    :return: design matrix as numpy array
    """
    return np.array([
        np.array([y(x) for y in functions]) for x in x_values
    ])
