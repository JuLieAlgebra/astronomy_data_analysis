"""
Galaxy classification by random forest with expert features.
"""
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
from . import galaxy_classification_tools as gc_tools


def generate_features_targets(data):
    """
    Parameters
    ----------
    data:       2D numpy array

    Returns
    -------
    features:   2D numpy array
    targets:    1D numpy array
    """
    targets = data['class']

    features = np.empty(shape=(len(data), 13))

    # color features
    features[:, 0] = data['u-g']
    features[:, 1] = data['g-r']
    features[:, 2] = data['r-i']
    features[:, 3] = data['i-z']

    # eccentricity (shape)
    features[:, 4] = data['ecc']

    # adaptive moments (shape)
    features[:, 5] = data['m4_u']
    features[:, 6] = data['m4_g']
    features[:, 7] = data['m4_r']
    features[:, 8] = data['m4_i']
    features[:, 9] = data['m4_z']

    # concentration in u filter (shape)
    features[:, 10] = data['petroR50_u']/data['petroR90_u']
    # concentration in r filter (shape)
    features[:, 11] = data['petroR50_r']/data['petroR90_r']
    # concentration in z filter (shape)
    features[:, 12] = data['petroR50_z']/data['petroR90_z']

    # removing null values from our data set
    null_mask = (features != -9999)
    row_mask = np.all(null_mask, axis=1)

    return features[row_mask], targets[row_mask]


def predict(data, n_estimators, kfold=10):
    """
    Creates Random Forest Classifier model, predicts with k-fold cross validation.
    User can specify number of decision trees and number of folds for cross validation.

    Parameters
    ----------
    data:         2D numpy array
    n_estimators: int
    kfold:        int

    Returns
    -------
    predictions:  TODO
    targets:      1D numpy array
    """
    # generate the features and targets
    features, targets = generate_features_targets(data)

    # instantiate a random forest classifier using n estimators
    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)

    # get predictions using 10-fold cross validation with cross_val_predict
    predictions = sklearn.model_selection.cross_val_predict(rfc, features, targets, kfold=10)

    # return the predictions and their actual classes
    return predictions, targets
