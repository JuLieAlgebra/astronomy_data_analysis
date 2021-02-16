"""
Tools for galaxy classification.
"""
import numpy as np
import itertools
from matplotlib import pyplot as plt


################ Metrics Functions ############################################

def calculate_accuracy(predicted, actual):
    """
    Parameters
    ----------
    predicted: 1D numpy array
    actual:    1D numpy array

    Returns
    -------
    accuracy:  float
    """
    accuracy = np.sum(actual == predicted) / len(predicted)
    return accuracy


################ Descriptive Statistics Functions ############################################

def data_stats(data, by_class=True):
    """
    TODO: implement not by class

    Returns mean and variance by class of the raw data

    returns 'elliptical' 'merger' 'spiral'

    Parameters
    ----------
    data:     2D numpy array
    by_class: boolean flag

    Returns
    -------
    class_std:  2D numpy array of shape (num_classes, num_features) so class_mean[0] gives class 0's stats for each feature
    class_mean: 2D numpy array of shape (num_classes, num_features) so class_mean[0] gives class 0's stats for each feature
    """
    if by_class:

        feature_names = data.dtype.names
        classes = np.unique(data['class'])
        print(classes)
        class_std = np.zeros((len(classes), len(feature_names) - 1))
        class_mean = np.zeros((len(classes), len(feature_names) - 1))

        for galaxy_type in np.arange(len(classes)):
            mask = (data['class'] == classes[galaxy_type])
            feature_std = np.zeros(len(feature_names) - 1)
            feature_mean = np.zeros(len(feature_names) - 1)

            for name, index in zip(feature_names[:-1], np.arange(len(feature_names) - 1)):
                null_mask = (data[name] != -9999)
                double_mask = (mask & null_mask)
                feature_std[index] = np.std(data[double_mask][name])
                feature_mean[index] = np.mean(data[double_mask][name])

            class_std[galaxy_type] = feature_std
            class_mean[galaxy_type] = feature_mean

        return class_std, class_mean


def feature_stats(features, targets=None):
    """
    If targets is provided, then will calculate the feature statistics by class instead of statistics of
    all of the features.

    Parameters
    ----------
    features:   2D numpy array of each data point with no null values for each feature
    targets:    1D numpy array of labels for each data point

    Returns
    -------
    standard deviation: 2D or 1D numpy array. If targets is provided, then is 2D np array with std[0] corresponding
                        to features statistics for class 0.

    mean:               Same as standard deviation.
    """
    if targets is not None:
        classes = np.unique(targets)

        # segmenting feature data by class
        merger_mask = (targets == 'merger')
        spiral_mask = (targets == 'spiral')
        elliptical_mask = (targets == 'elliptical')

        # std calculations
        merger_std = np.std(features[merger_mask], axis=0)
        spiral_std = np.std(features[spiral_mask], axis=0)
        elliptical_std = np.std(features[elliptical_mask], axis=0)

        # mean calculations
        merger_mean = np.mean(features[merger_mask], axis=0)
        spiral_mean = np.mean(features[spiral_mask], axis=0)
        elliptical_mean = np.mean(features[elliptical_mask], axis=0)

        return np.vstack((spiral_std, merger_std, elliptical_std)), np.vstack((merger_mean, spiral_mean, elliptical_mean))

    else:
        return np.std(features, axis=0), np.mean(features, axis=0)


################ Plotting Functions ############################################

def plot_stats(xdata, stats, labels=None, xlabels=None, ylabel=None, title=None):
    """
    Stats must be a list of multiple sets of stats
    Plots a bar chart

    Parameters
    ----------
    xdata:   data for the x axis, in this case, each feature
    stats:   list of statistics - each statistic should have a value for each feature
    labels:  list of strings for the classes
    xlabels: names of each feature
    ylabel:  string flag for plotting std or mean of features
    """
    fig = plt.figure()

    # calculating an appropriate spacing for the bar chart
    width = len(xdata) / (len(stats[0])*len(stats)*1.2)

    if xlabels is not None:
        plt.xticks(range(len(xdata)), xlabels)

    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

    for index, stat in enumerate(stats):
        if labels is not None:
            plt.bar(xdata + index*width, stat, width=width, label=labels[index])
        else:
            plt.bar(xdata + index*width, stat, width=width)

    plt.legend()
    plt.xlabel("Feature Number")

    if ylabel == 'std':
        plt.ylabel("Standard Deviation in Each Feature")
        plt.title("Standard Deviation by Feature")
    elif ylabel == 'mean':
        plt.ylabel("Mean in Each Feature")
        plt.title("Mean by Feature")
    elif ylabel is not None:
        plt.ylabel(ylabel)
        plt.title(title)
    fig.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Code courtesy of Dr. Tara Murphy.

    Parameters
    ----------
    cm:        confusion matrix
    classes:   list of unique classes
    normalize: boolean flag
    cmap:      color choice for plot
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    fig.show()