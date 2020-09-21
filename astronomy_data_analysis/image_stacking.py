"""
Here we only have functions used for image stacking.
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


def scalable_mean_fits(data_stack):
    """
    A mean function that only needs to load one image into memory at a time.

    Parameters
    ----------
    data_stack: 3D numpy array of fits image data. See fits_tools.get_data_stack.
    """
    hist = np.zeros(data_stack[0].shape)
    for image in data_stack:
        hist += image

    mean = hist / len(data_stack)
    return mean


def median_fits(data_stack):
    """
    Returns the median value across the same element in each fits files.
    Return has the same shape as one image in data_stack.

    Parameters
    ----------
    data_stack: 3D numpy array of fits image data. See fits_tools.get_data_stack.
    """
    return np.median(data_stack, axis=0)


def scalable_median_fits(data_stack, num_bins=5):
    """
    Implements numpythonic binapprox algorithm
    For more details: http://www.stat.cmu.edu/~ryantibs/median/

    TODO fix the data stack stuff, use scalable mean and std for computing stats in hist

    Parameters
    ----------
    file_list: blad
    num_bins: int

    """
    mean, std, left_bins, bins = median_histogram(data_stack, num_bins)
    mid = (len(data_stack) + 1) / 2 # we only want to count until we've reached the median value

    bin_width = 2 * std / num_bins

    cumsumm = np.cumsum(np.dstack((left_bins, bins)), axis=2)
    b = np.argmax(cumsumm >= mid, axis=2) - 1

    median = mean - std + bin_width*(b + 0.5)
    return median


def median_histogram(data_stack, num_bins=5):
    """
    Constructs the histogram for scalable_median_fits. End result will have a histogram for
    each "pixel" of the fits image.

    Parameters
    ----------
    file_list: blad
    num_bins: int


    Returns
    -------
    mean:   2D numpy array with dimensions = input image. Mean image of all of the data.
    std:    2D numpy array with dimensions = input image. Std of all of the data.
    left_bins: 2D numpy array with dimensions = input image. Number of pixels who were more than 1 std from mean.
    bins:   3D numpy array with dimensions = (input image shape, num_bins). Histogram.
    """
    mean = mean_fits(data_stack)
    std = np.std(data_stack, axis=0)

    minval = mean - std
    maxval = mean + std
    bin_width = 2 * std / num_bins

    bins = np.zeros((data_stack[0].shape[0], data_stack[0].shape[1], num_bins)) # , dtype=int TODO think about this
    # we are going to count the number of times we see a value less than mean - std.
    left_bins = np.zeros(data_stack[0].shape)

    for i, data in enumerate(data_stack):
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
