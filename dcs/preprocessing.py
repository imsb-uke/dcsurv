from dcs import defaults
import sklearn.model_selection
# import numpy as np
# import pandas as pd
# from sklearn.pipeline import Pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def split_X_y(df, cols_target=defaults.LABELS, index="patient_id"):
    """
    Splits df in X and y with y being vector of tuples of target cols
    """

    X = df.drop(cols_target, axis=1)
    y = df[cols_target]

    return X, y


def train_test_split(*args, **kwargs):
    return sklearn.model_selection.train_test_split(*args, **kwargs)


def train_test_split_X_y(df, labels=defaults.LABELS, **train_test_split_kwargs):
    train, test = train_test_split(df, **train_test_split_kwargs)
    train_X, train_y = split_X_y(train, labels)
    test_X, test_y = split_X_y(test, labels)

    return train_X, train_y, test_X, test_y
