from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import openml

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LinRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LinearRegression) -> List:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model, params
):
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


# def set_initial_params(model: LogisticRegression):
#     """Sets initial parameters as zeros Required since model params are
#     uninitialized until model.fit is called.
#
#     But server asks for initial parameters from clients at launch. Refer
#     to sklearn.linear_model.LogisticRegression documentation for more
#     information.
#     """
#     n_classes = 10  # MNIST has 10 classes
#     n_features = 784  # Number of features in dataset
#     model.classes_ = np.array([i for i in range(10)])
#
#     model.coef_ = np.zeros((n_classes, n_features))
#     if model.fit_intercept:
#         model.intercept_ = np.zeros((n_classes,))


# def load_mnist() -> Dataset:
#     """Loads the MNIST dataset using OpenML.
#
#     OpenML dataset link: https://www.openml.org/d/554
#     """
#     mnist_openml = openml.datasets.get_dataset(554)
#     Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
#     X = Xy[:, :-1]  # the last column contains labels
#     y = Xy[:, -1]
#     # First 60000 samples consist of the train set
#     x_train, y_train = X[:60000], y[:60000]
#     x_test, y_test = X[60000:], y[60000:]
#     return (x_train, y_train), (x_test, y_test)


def create_sklearn_model():
    model = LinearRegression()
    return model

import numpy as np
from collections import Counter


def load_fever() -> Dataset:
    """Loads the MNIST dataset using OpenML.

    OpenML dataset link: https://www.openml.org/d/554
    """
    # mnist_openml = openml.datasets.get_dataset(554)
    # Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    # X = Xy[:, :-1]  # the last column contains labels
    # y = Xy[:, -1]
    # First 60000 samples consist of the train set

    fever = pd.read_csv('sklearn_fever/fever.csv')
    print(fever.columns)
    feature_names = ['trip_distance(km)','city','motor_way','country_roads','month']
    fever = fever[feature_names + ['quantity(kWh)']]
    fever = fever.dropna(axis=0, how='any')
    fever = fever.reset_index(drop=True)
    print(fever.shape)
    y = np.array(fever['quantity(kWh)'])
    X = np.array(fever[feature_names])
    x_train, y_train = X[:15000], y[:15000]
    x_test, y_test = X[15000:], y[15000:]
    print(x_train.shape)
    print(type(x_train))
    print(y_train.shape)
    return (x_train, y_train), (x_test, y_test)
def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )