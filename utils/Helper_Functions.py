"""
Essential preprocessing and utility functions for CATHODE analysis.
LogitScaler from https://github.com/uhh-pd-ml/sk_cathode/blob/main/sk_cathode/utils/preprocessing.py
"""
import os
import subprocess
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.special import logit, expit
from numbers import Real

class LogitScaler(MinMaxScaler):
    """Preprocessing scaler that performs a logit transformation on top
    of the sklean MinMaxScaler. It scales to a range [0+epsilon, 1-epsilon]
    before applying the logit. Setting a small finite epsilon avoids
    features being mapped to exactly 0 and 1 before the logit is applied.
    If the logit does encounter values beyond (0, 1), it outputs nan for
    these values.
    """

    _parameter_constraints: dict = {
        "epsilon": [Real],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, epsilon=0, copy=True, clip=False):
        self.epsilon = epsilon
        self.copy = copy
        self.clip = clip
        super().__init__(feature_range=(0+epsilon, 1-epsilon),
                         copy=copy, clip=clip)

    def fit(self, X, y=None):
        super().fit(X, y)
        return self

    def transform(self, X):
        z = logit(super().transform(X))
        z[np.isinf(z)] = np.nan
        return z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return super().inverse_transform(expit(X))

    def jacobian_determinant(self, X):
        z = super().transform(X)
        return np.prod(z * (1 - z), axis=1, keepdims=True
                       ) / np.prod(self.scale_)

    def log_jacobian_determinant(self, X):
        z = super().transform(X)
        return np.sum(np.log(z * (1 - z)), axis=1, keepdims=True
                      ) - np.sum(np.log(self.scale_))

    def inverse_jacobian_determinant(self, X):
        z = expit(X)
        return np.prod(z * (1 - z), axis=1, keepdims=True
                       ) * np.prod(self.scale_)

    def log_inverse_jacobian_determinant(self, X):
        z = expit(X)
        return np.sum(np.log(z * (1 - z)), axis=1, keepdims=True
                      ) + np.sum(np.log(self.scale_))


def split_dictionary(dictionary, split_size):
    """
    Split a dictionary into smaller dictionaries based on a specified size.
    
    Parameters:
    -----------
    dictionary : dict
        The dictionary to be split.
    split_size : int
        The approximate size for each split.
        
    Returns:
    --------
    list : List of dictionaries.
    """
    items = list(dictionary.items())
    length = len(items)
    num_parts = max(1, length // split_size)
    chunk_size = (length + num_parts - 1) // num_parts
    
    parts = []
    for i in range(0, length, chunk_size):
        part = dict(items[i:i + chunk_size])
        parts.append(part)
    
    return parts