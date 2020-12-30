"""
Some library functions for handling fits files, the most common
data file type in astronomy.

Fits files include a lot of data, including an image and header file.
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def get_fits_data(filename):
    """
    Returns image data (2D numpy array) of file

    Parameters
    ----------
    filename: string of fits file name including path to object
    """
    if filename[-5:] != '.fits':
        filename += filename + '.fits'

    # opening file and extracting data
    hdulist = fits.open(filename)
    data = hdulist[0].data.copy()
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
    fig = plt.figure()
    ax = fig.add_axes()
    plt.imshow(image.T, cmap=plt.cm.viridis)
    plt.xlabel('x-pixels (RA)')
    plt.ylabel('y-pixels (Dec)')
    plt.colorbar()
    if title is not None:
         plt.title(title)
    fig.show()


def plot_images(data_stack, title='', ncols=4, subtitles=None):
    """
    Takes in a data stack. See fits_tools.get_data_stack.
    Plots all fits data in same figure.

    Parameters
    ----------
    data_stack: 3D numpy array
    title:      string, optional
    ncols:      int, optional
    subtitles:  list of strings, with subtitle for each image

    TODO: Make nicer.
    """
    nrows = data_stack.shape[0] // ncols + (data_stack.shape[0] % ncols) # array of sub-plots
    figsize = [6, 8]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        # Leaving here for future work
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        if i < data_stack.shape[0]:
            axi.imshow(data_stack[i])
            if subtitles:
                axi.title.set_text(subtitles[i])

    fig.suptitle(title)
    plt.tight_layout(True)
    plt.show()


def plot3D(cartesian_catalog, highlighted=None, title=None):
    """
    Takes in numpy array of star coordinates - must be what kd_cross_matching.ra_dec_to_cartesian
    returns.
    Highlighted will plot those points (subset of cartesian catalog) in another color

    Parameters
    ----------
    cartesian_catalog:  2D numpy array with first three columns as cartesian coordinates in radians
    highlighted:        1D numpy array of indices/star IDs to be highlighted
    title:              string, optional
    """
    coor = cartesian_catalog
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X = coor.T[0]
    Y = coor.T[1]
    Z = coor.T[2]
    c = X + Y
    ax.scatter(X, Y, Z, c=c)

    # TODO think about whether to add highlighted point option
    #highlight_points = coor[:, :, :, np.int(highlighted.T)].T
    #ax.scatter(highlight_points[0], highlight_points[1], highlight_points[2], c='r')

    # Create cubic bounding box to simulate equal aspect ratio, thank you stackoverflow user
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.grid()
    ax.set_title(title)
    plt.show()