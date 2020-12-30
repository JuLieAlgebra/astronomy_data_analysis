"""
Galaxy classification by expert features.
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

    Returns
    -------
    """
    # complete the function by calculating the concentrations

    targets = data['class']

    features = np.empty(shape=(len(data), 13))
    # color features
    features[:, 0] = data['u-g']
    features[:, 1] = data['g-r']
    features[:, 2] = data['r-i']
    features[:, 3] = data['i-z']
    # eccentricity
    features[:, 4] = data['ecc']
    # adaptive moments
    features[:, 5] = data['m4_u']
    features[:, 6] = data['m4_g']
    features[:, 7] = data['m4_r']
    features[:, 8] = data['m4_i']
    features[:, 9] = data['m4_z']

    # concentration in u filter
    features[:, 10] = data['petroR50_u']/data['petroR90_u']
    # concentration in r filter
    features[:, 11] = data['petroR50_r']/data['petroR90_r']
    # concentration in z filter
    features[:, 12] = data['petroR50_z']/data['petroR90_z']

    # removing null values from our data set
    null_mask = (features != -9999)
    row_mask = np.all(null_mask, axis=1)

    return features[row_mask], targets[row_mask]


def predict(data, n_estimators):
    """
    Creates model, predicts with cross fold validation

    Parameters
    ----------
    data:         test
    n_estimators: test

    Returns
    -------
    predictions:  test
    targets:      test
    """
    # generate the features and targets
    features, targets = generate_features_targets(data)

    # instantiate a random forest classifier using n estimators
    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)

    # get predictions using 10-fold cross validation with cross_val_predict
    predictions = sklearn.model_selection.cross_val_predict(rfc, features, targets, cv=10)

    # return the predictions and their actual classes
    return predictions, targets
