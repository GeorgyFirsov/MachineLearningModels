from typing import Tuple, Optional
from itertools import chain
from random import shuffle, seed

import numpy as np


def train_test_split(
        x_values: np.ndarray, y_values: np.ndarray, *, test_part: float = 0.2, random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits received x-values and y_values into 4 sets - train and test
    samples for X and corresponding samples for Y.

    :param x_values: x-values
    :param y_values: corresponding y-values
    :param test_part: ratio len(y_test) / len(y_values) (only named parameter)
    :param random_seed: random seed (only named parameter)
    :return: 4 samples - tests and trains for X and Y
    """

    # test_part must be in closed interval [0, 1]
    if test_part > 1 or test_part < 0:
        raise ValueError("test_part must be in closed interval[0, 1]")

    # Copy values to prevent corruption
    internal_x = x_values
    internal_y = y_values

    multiple_features = False
    wrong_length = False

    # Check if X and Y have equal length
    if len(internal_x) != len(internal_y):
        if not isinstance(internal_x[0], np.ndarray):
            wrong_length = True
        else:
            for x in internal_x:
                if len(x) != len(y_values):
                    wrong_length = True
            multiple_features = True

    if wrong_length:
        raise ValueError("X and Y values' iterables must be the same length")

    # Set random seed if specified
    if random_seed is not None:
        seed(random_seed)

    if multiple_features:
        internal_x = internal_x.T

    test_count = int(len(internal_x) * test_part)
    train_count = len(internal_x) - test_count

    # Generate flags list, where 1 on i-th position means that
    # x_values[i] and y_values[i] should go to x_test and y_test respectively
    indices = list(
        chain([1 for _ in range(test_count)], [0 for _ in range(train_count)])
    )
    shuffle(indices)

    # Roll back random seed not to harm external code
    seed()

    x_train = list()
    x_test = list()
    y_train = list()
    y_test = list()

    # Split values into certain lists
    for x, y, i in zip(internal_x, internal_y, indices):
        if i == 1:
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x)
            y_train.append(y)

    x_train = np.array(x_train).T if multiple_features else np.array(x_train)
    x_test = np.array(x_test).T if multiple_features else np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


def make_features_pack(*separate_features) -> np.ndarray:
    """Makes a pack from separate features as on
    following pattern: [[][][]...[]]

    :param separate_features: distinct features
    :return: features pack
    """
    return np.array([
        np.array([x for x in feature]) for feature in separate_features
    ])
