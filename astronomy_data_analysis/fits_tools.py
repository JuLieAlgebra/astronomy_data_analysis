"""
Some library functions for handling fits files, the most common
data file type in astronomy.

Fits files include a lot of data, including an image and header file.
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


##### HELPER FUNCTIONS FOR LOADING FITS FILES AND STACKING #####

def get_fits_data(filename):
    """
    Returns image data (2D numpy array) of file

    Parameters
    ----------
    filename: string of fits file name
    """
    if filename[-5:] != '.fits':
        filename += filename + '.fits'

    # opening file and extracting data
    hdulist = fits.open(filename)
    data = hdulist[0].data
    hdulist.close()

    return data


def get_data_stack(file_list):
    """
    Takes in list of fits files
    Each fits file must have image of the same shape

    First element of the data stack indexes into a fits image
    Think of the data stack as a literal stack of papers, with
    each paper as a fits image. Computing along axis 0 will go
    pixel-wise through each image, like an arrow stabbing the
    stack of papers.

    Returns 3D "cube" of fits image data
    If you only have one fits file, use get_fits_data instead

    Parameters
    ----------
    file_list: list of strings of fits file names
    """
    data_slice_shape = np.shape(get_fits_data(file_list[0]))
    data_stack = np.zeros((len(file_list), data_slice_shape[0], data_slice_shape[1]))

    for index, file in enumerate(file_list):
         data_stack[index] = get_fits_data(file)

    return data_stack


############ PLOTTING FUNCTIONS ##############

def plot(image, title=None):
    """
    Plots data of fits file.
    User must pass in data of fits image (hdulist[0].data)

    Parameters
    ----------
    image: 2D numpy array
    title: string, optional
    """
    plt.imshow(image.T, cmap=plt.cm.viridis)
    plt.xlabel('x-pixels (RA)')
    plt.ylabel('y-pixels (Dec)')
    plt.colorbar()
    if title is not None:
         plt.title(title)
    plt.show()


def plot_images(data_stack, title='', ncols=4, subtitles=None):
    """
    Takes in a data stack. See fits_tools.get_data_stack.
    Plots all fits data in same figure.

    Parameters
    ----------
    data_stack: 3D numpy array
    ncols: int, optional
    subtitles: list of strings, with subtitle for each image

    TODO: Make nicer.
    """
    nrows = data_stack.shape[0] // ncols + int(bool(data_stack.shape[0] % ncols)) # array of sub-plots
    figsize = [6, 8]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        if i < data_stack.shape[0]:
            axi.imshow(data_stack[i])
            if subtitles:
                axi.title.set_text(subtitles[i])

    fig.suptitle(title)
    plt.tight_layout(True)
    plt.show()
