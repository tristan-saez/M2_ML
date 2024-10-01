"""

This file is dedicated to SVM algorithm (from scratch) without kernel usage.

"""
import pandas as pd


def svm(data):
    """

    main function for SVM usage without kernel
    :param data: database
    """
    params = {
        "l_rate": 0.01,
        "nb_iter": 100,
        "weight": 0,
        "tradeoff": 0.01,
        "intercept": 0
    }

    fit_svm(data, data, params)


def fit_svm(x_train, y_train, params):
    pass


def test_svm():
    pass


def plot_svm():
    pass
