"""
Some functions used for image stacking - including scalable mean and median algorithms
for large data sets.
"""
import numpy as np
from astropy.io import fits
from . import fits_tools
import matplotlib.pyplot as plt


def mean_fits(data_stack):
    """
    Returns the mean value across the same element in each fits files.
    Return has the same shape as one image in data_stack.

    Parameters
    ----------
    data_stack: 3D numpy array of fits image data. See fits_tools.get_data_stack.
    """
    return np.mean(data_stack, axis=0)


def std_fits(data_stack):
    """
    Returns the std value across the same element in each fits files.
    Return has the same shape as one image in data_stack.

    Parameters
    ----------
    data_stack: 3D numpy array of fits image data. See fits_tools.get_data_stack.
    """
    return np.std(data_stack, axis=0)


def median_fits(data_stack):
    """
    Returns the median value across the same element in each fits files.
    Return has the same shape as one image in data_stack.

    Parameters
    ----------
    data_stack: 3D numpy array of fits image data. See fits_tools.get_data_stack.
    """
    return np.median(data_stack, axis=0)


def scalable_stats_fits(file_list):
    """
    A naive mean, standard deviation function that only needs to load one image into memory at a time.

    TODO: implement Welford's method instead, which will have much more numeric stability as the number
    of images increases dramatically. However, for less than ~ a quarter million images, this naive
    implementation will still do just fine.

    Parameters
    ----------
    file_list: list of string fits filenames
    """
    dim = fits_tools.get_fits_data(file_list[0]).shape
    hist = np.zeros(dim)
    sq_hist = np.zeros(dim)

    for file in file_list:
        data = fits_tools.get_fits_data(file)
        hist += data
        sq_hist += data * data

    mean = hist / len(file_list)
    std = np.sqrt(sq_hist / len(file_list) - mean * mean)
    return mean, std


def scalable_median_fits(file_list, num_bins=5):
    """
    Implements numpythonic binapprox algorithm, which runs in O(number of images).
    For more details: http://www.stat.cmu.edu/~ryantibs/median/

    Parameters
    ----------
    file_list: list of string fits filenames
    num_bins: int
    """
    mean, std, left_bins, bins = median_histogram(file_list, num_bins)
    midpoint = (len(file_list) + 1) / 2 # we only want to count until we've reached the median value

    bin_width = 2 * std / num_bins

    # All the heavy duty lifting happens in these two lines!
    cumsumm = np.cumsum(np.dstack((left_bins, bins)), axis=2) # we want to integrate our histogram until we reach the median value
    b = np.argmax(cumsumm >= midpoint, axis=2) - 1 # argmax returns the first instance of true it finds

    # once we've found the bins in the histograms with the median values in it, compute median
    median = mean - std + bin_width*(b + 0.5)
    return median


def median_histogram(file_list, num_bins=5):
    """
    Constructs the histogram for scalable_median_fits. End result will have a histogram for
    each "pixel" of the fits image.

    Parameters
    ----------
    file_list: list of string fits filenames
    num_bins: int


    Returns
    -------
    mean:   2D numpy array with dimensions = input image. Mean image of all of the data.
    std:    2D numpy array with dimensions = input image. Std of all of the data.
    left_bins: 2D numpy array with dimensions = input image.
               Number of pixels who were more than 1 std from mean.
    bins:   3D numpy array with dimensions = (input image shape, num_bins). Histogram.
    """
    mean, std = scalable_stats_fits(file_list)

    minval = mean - std
    maxval = mean + std
    bin_width = 2 * std / num_bins

    image_dim = fits_tools.get_fits_data(file_list[0]).shape
    bins = np.zeros((image_dim[0], image_dim[1], num_bins)) # , dtype=int TODO think about this
    # we are going to count the number of times we see a value less than mean - std.
    left_bins = np.zeros(image_dim)

    for i, file in enumerate(file_list):
        data = fits_tools.get_fits_data(file)

        low_values = (data < minval)
        left_bins[low_values] +=1

        in_range = (data >= minval) & (data < maxval)
        bin_val = (data - minval) / bin_width

        bin_index = np.array((data[in_range] - (minval[in_range])) / bin_width[in_range], dtype=int)
        bins[in_range, bin_index] += 1
    return mean, std, left_bins, bins


def brightest_pixel(image):
    """
    Takes data stack of fits image data.
    Returns the pixel (array like) value where the image is brightest.

    Parameters
    ----------
    image: 2D numpy array or 3D numpy array of image data. See fits_tools.get_data_stack.
    """
    # find the pixel with the brightest value in the image
    unraveled_location = np.argmax(np.array(image)) # argmax returns index in flattened version of array
    location = np.unravel_index(unraveled_location, np.shape(image)) # get back our location in our 2D image

    return location
