"""
Here we only have functions used for image stacking.
"""
import numpy as np
from astropy.io import fits
from . import fits_tools

def mean_fits(data_stack):
    """
    Takes in list of fits files
    Each fits file must have image of the same shape
    Returns the mean value across the same element in each fits files
    shape of return is same as shape of each image file

    Parameters
    ----------
    data_stack: 3D numpy array of fits image data. See fits_tools.get_data_stack.
    """
    return np.mean(data_stack, axis = 0)


def brightest_pixel(image):
    """
    Takes data stack of fits image data
    Returns the pixel (array like) value where the image is brightest

    Parameters
    ----------
    image: 2D numpy array or 3D numpy array of image data. See fits_tools.get_data_stack.
    """
    # find the pixel with the brightest value in the image
    unraveled_location = np.argmax(np.array(image)) # argmax returns index in flattened version of array
    location = np.unravel_index(unraveled_location, np.shape(image)) # get back our location in our 2D image

    return location


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

