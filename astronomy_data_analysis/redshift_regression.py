"""
Creating photometric redshifts with machine learning.

NOTE:   Left features normalized, so scoring is very very small because features are
        small valued now

TODO
    - fix plot by color indices func
    - add demonstration to ipython notebook
    - finish writing unit tests for functions
    - box plot
    - residuals
    - finish documentation, clean up code (possibly refactor)
    - deal with the "catastrophic errors" mentioned in scikit learn post
    - rework models for the discrete class
    - monetary estimates for this over spectroscopic redshift measurements
    - risk analysis for accepting photometric redshift over spectroscopic
 """
import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import itertools

from sklearn.linear_model import LinearRegression


############### Plotting ##################


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
    width = len(xdata) / (len(stats[0]) * len(stats) * 1.2)

    if xlabels is not None:
        plt.xticks(range(len(xdata)), xlabels)

    plt.grid(color="#95a5a6", linestyle="--", linewidth=2, axis="y", alpha=0.7)

    for index, stat in enumerate(stats):
        if labels is not None:
            plt.bar(xdata + index * width, stat, width=width, label=labels[index])
        else:
            plt.bar(xdata + index * width, stat, width=width)

    plt.legend()
    plt.xlabel("Feature Number")

    if ylabel == "std":
        plt.ylabel("Standard Deviation in Each Feature")
        plt.title("Standard Deviation by Feature")
    elif ylabel == "mean":
        plt.ylabel("Mean in Each Feature")
        plt.title("Mean by Feature")
    elif ylabel is not None:
        plt.ylabel(ylabel)
        plt.title(title)
    elif title is not None:
        plt.title(title)
    plt.show()


def plot_correlation_matrix(
    cov_m, std, classes, normalize=False, title="Correlation Matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the correlation matrix.
    Normalization can be applied by setting `normalize=True`.

    Code courtesy of Dr. Tara Murphy.

    Parameters
    ----------
    cov_m:        covariance matrix
    classes:   list of unique classes
    normalize: boolean flag
    cmap:      color choice for plot
    """
    # fig = plt.figure()
    cm = cov_m
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cm[i, j] = np.round(cm[i, j] / (std[i] * std[j]), 3)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized correlation matrix")
    else:
        print("Correlation matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("Features, including Label")
    plt.xlabel("Features, including Label")
    plt.show()


def plot_boxplot(data, labels, title=None):
    """ """
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.ylabel("Features, including Label")
    plt.xlabel("Features, including Label")
    plt.show()


def plot_galaxies_by_color_indices(data, labels, features, name="Default.png"):
    """
    Not working.
    """
    # 6 for all the possible parings
    plt.figure(figsize=(15, 10))
    batch = 1
    for ind, subset in enumerate(itertools.combinations(labels, 2)):
        print(subset[0])
        subplt = plt.subplot(int("23%d" % (ind + 1)))
        subplt.scatter(
            features[subset[0]].values[::batch],
            features[subset[1]].values[::batch],
            marker=".",
        )
        subplt.set_title("Scatter plot for %s - %s" % (subset[0], subset[1]))
        subplt.set_xlabel("%s" % subset[0])
        subplt.set_ylabel("%s" % subset[1])
    plt.show()


#################### Modeling ############################


def generate_features_targets(data, normalize=True):
    """
    Note:   This engineers some features, called 'color' in astronomy.
            Common in the field.

    Parameters
    ----------
    data:       numpy structured array (see numpy docs for details)

    Returns
    -------
    features:   2D numpy array
    targets:    2D numpy array

    """
    feature_names = ["u - g", "g - r", "r - i", "i - z", "spec_class", "redshift"]
    features = np.empty(shape=(len(data), len(feature_names)))

    # color features
    features[:, 0] = data["u"] - data["g"]
    features[:, 1] = data["g"] - data["r"]
    features[:, 2] = data["r"] - data["i"]
    features[:, 3] = data["i"] - data["z"]

    # embedding for the different object classes
    classes, counts = np.unique(data["spec_class"], return_counts=True)
    print("This many galaxies: {}, this many quasars: {}".format(counts[0], counts[1]))

    # Quasar is Class -1, Galaxy is class +1
    features[:, 4] = 2 * np.float64(data["spec_class"] == classes[0]) - 1
    print("classes ", classes)

    # Redshift to predict
    features[:, 5] = data["redshift"]

    # removing null values from our data set
    valid_mask = features != -9999
    row_mask = np.all(valid_mask, axis=1)

    # assert len(features[features == 'None']) == 0
    # assert len(features[features == 'Null']) == 0
    # assert len(features[features == -9999]) == 0

    if normalize:
        # normalize features
        means = np.mean(features, axis=0)
        features = (features - means) / len(features)

    return features[row_mask], feature_names


def split(data, fraction):
    """
    Shuffles and splits data according to fraction. Assumes last column is the targets.

    Parameters
    ----------
    features: 2D numpy array
    targets:  2D numpy array
    fraction: float

    Returns
    -------
    f_train: 2D numpy array
    t_train: 2D numpy array
    f_test:  2D numpy array
    t_test:  2D numpy array
    """
    # Shuffle & Split
    shuffled_data = np.random.shuffle(data)
    data_train = data[0 : int(fraction * len(data))]
    data_test = data[int(fraction * len(data)) :]

    # Training Set
    f_train = data_train[:, :-1]  # features
    t_train = data_train[:, -1]  # targets

    # Test Set
    f_test = data_test[:, :-1]  # features
    t_test = data_test[:, -1]  # targets

    return f_train, t_train, f_test, t_test


def train(features, targets):
    """
    Setup, compile, and train a sequential Keras model

    Parameters
    ----------
    features: 2D numpy array
    targets:  2D numpy array

    Returns
    -------
    model: tensorflow sequential model object
    """
    model = keras.Sequential(
        [
            # keras.layers.Dense(units=100, activation="tanh"),
            # keras.layers.Dense(units=20, activation="tanh"),
            # keras.layers.Dense(units=5, activation="tanh"),
            keras.layers.Dense(1)
        ]
    )
    model.compile(
        loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"]
    )

    model.fit(features, targets)

    return model


def linearRegression(features, targets):
    """
    Scipy machine learning algorithm example

    Parameters
    ----------
    features: 2D numpy array
    targets:  2D numpy array
    """
    reg = LinearRegression().fit(features, targets)

    return reg


def predict(model, test_data):
    """
    Parameters
    ----------
    model:      tensorflow sequential model object
    test_data:  2D numpy array
    """
    return model.predict(test_data)


#################### Metrics ############################


def median_diff(predicted, actual):
    """
    Parameters
    ----------
    predicted: 1D numpy array
    actual:    1D numpy array
    """
    return np.median(np.abs(predicted - actual))


def mse(predicted, actual):
    """
    Parameters
    ----------
    predicted: 1D numpy array
    actual:    1D numpy array
    """
    print(predicted.shape, actual.shape)
    return np.mean(np.square(predicted - actual))


#################### Testing ############################

if __name__ == "__main__":
    parent_directory = os.getcwd()
    path = os.path.join(
        parent_directory, os.pardir, "sample_data", "galaxy_catalogs", ""
    )

    raw = np.load(path + "sdss_galaxy_colors.npy")
    bands = np.column_stack((raw["u"], raw["g"], raw["r"], raw["i"], raw["z"]))
    print(bands.shape)
    print(bands[:5])
    print(raw["u"].shape)
    raw_cov = np.cov(bands.T)
    raw_std = np.std(bands, axis=0)
    # plot_correlation_matrix(raw_cov, raw_std, raw.dtype.names[0:5], title="Correlation Matrix of Raw Data")
    print(raw[0:3])
    print(raw.dtype.names)

    data, feature_names = generate_features_targets(raw)
    # testing correlations between columns
    mean = np.mean(data, axis=0)  # mean for each feature
    std = np.std(data, axis=0)  # std for each feature
    cov = np.cov(data.T)  # cov for the features

    print("mean: ", mean)
    print("Stats ")
    print(mean.shape)
    print(std.shape)
    print(cov.shape)
    print()

    # plot_stats(np.arange(6), np.vstack((mean, std)), labels=['mean', 'std'], xlabels=feature_names, title='Mean and Std of Features')
    # plot_galaxies_by_color_indices(data, f_train, feature_names)
    # plot_correlation_matrix(cov, std, feature_names, title="Correlation Matrix of Features")
    # plot_boxplot(data, feature_names)

    # testing correlations between columns
    mean = np.mean(data, axis=0)  # mean for each feature
    std = np.std(data, axis=0)  # std for each feature
    cov = np.cov(data.T)  # cov for the features

    print("mean: ", mean)
    print("Stats ")
    print(mean.shape)
    print(std.shape)
    print(cov.shape)
    print()

    f_train, t_train, f_test, t_test = split(data, fraction=0.8)
    # plot_galaxies_by_color_indices(data, f_train, feature_names)

    model = train(f_train, t_train)
    # print(model.summary())
    print()

    print("Predictions on Training Set")
    sample = len(f_test)
    NN_predictions = model.predict(f_train[0:sample])
    print(np.sqrt(mse(NN_predictions, t_train[0:sample])))
    print(median_diff(NN_predictions, t_train[0:sample]))
    print()

    print("Predictions on Test Set")
    NN_predictions = model.predict(f_test)
    print("mean diff: ", np.sqrt(mse(NN_predictions, t_test)))
    print("median diff: ", median_diff(NN_predictions, t_test))
    print()
    # print("model parameters: ", model.layers[0].get_weights()[0])

    print("Linear Regression Model and Scoring")
    lin = linearRegression(f_train, t_train)
    lin_predictions = lin.predict(f_test)
    print("mean diff: ", np.sqrt(mse(lin_predictions, t_test)))
    print("median diff: ", median_diff(lin_predictions, t_test))
    # print("model parameters: ", lin.coef_, lin.intercept_)
    # print(lin.score(f_test, t_test))

    importance = lin.coef_
    # summarize feature importance
    for i, v in enumerate(importance):
        print("Feature: %0d, Weight: %.5f" % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()
