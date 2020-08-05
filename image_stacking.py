#!/usr/bin/env python3
"""  Small script to perform some analysis on FITS files, which
     is the data type most commonly used by astronomers.

     author: Julieanna Bacon
"""
import sys # to read in command line arguments
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


##### HELPER FUNCTIONS FOR LOADING FITS FILES AND STACKING #####

def get_fits_data(filename):
     """  Takes string filename of a fits file in
          Returns image data (2D numpy array) of file
     """
     if filename[-5:] != '.fits':
       filename += filename + '.fits'

     # opening file and extracting data
     hdulist = fits.open(filename)
     data = hdulist[0].data
     hdulist.close()

     return data


def get_data_stack(file_list):
     """  Takes in list of fits files
          Each fits file must have image of the same shape

          First element of the data stack indexes into a fits image
          Think of the data stack as a literal stack of papers, with
          each paper as a fits image. Computing along axis 0 will go
          pixel-wise through each image, like an arrow stabbing the
          stack of papers.

          Returns 3D "cube" of fits image data
          If you only have one fits file, use get_fits_data instead
     """
     # setting up the right shape for our stack of data to then be filled in
     data_slice_shape = np.shape(get_fits_data(file_list[0]))
     data_stack = np.zeros((len(file_list), data_slice_shape[0], data_slice_shape[1]))

     for index, file in enumerate(file_list):
          data_stack[index] = get_fits_data(file)

     return data_stack


############ PLOTTING FUNCTIONS ##############

def plot(image):
     """ Plots data of fits file.
         User must pass in data of fits image (hdulist[0].data)
     """
     import matplotlib.pyplot as plt
     plt.imshow(image.T, cmap=plt.cm.viridis)
     plt.xlabel('x-pixels (RA)')
     plt.ylabel('y-pixels (Dec)')
     plt.colorbar()
     plt.show()


def plot_images(images):
     """  Takes in a 1D array of fits image data (hdulist[0].data)
          Plots all fits data in same figure
     """
     # To be implemented
     pass


##############################################

def mean_fits(file_list):
     """  Takes in list of fits files
          Each fits file must have image of the same shape
          Returns the mean value across the same element in each fits files
          shape of return is same as shape of each image file
     """
     data_stack = get_data_stack(file_list)

     return np.mean(data_stack, axis = 0)


def brightest_pixel(image):
     """ Takes in fits image data
         Returns the pixel (array like) value where the image is
         brightest
     """
     # find the pixel with the brightest value in the image
     unraveled_location = np.argmax(np.array(image)) # argmax returns index in flattened version of array
     location = np.unravel_index(unraveled_location, np.shape(image)) # get back our location in our 2D image

     return location


################################################

if __name__ == '__main__':

     path = '/home/bacon/code/datasets/astronomy_data/pulsar images/'
     if len(sys.argv) > 1:
          fits_file = sys.argv[1]
     else:
          fits_file = path + '0000.fits'

     image = get_fits_data(fits_file)
     print(image)
     plot(image)
     quit()

     file_list = [ path + '000{0}.fits'.format(i) for i in range(0, 9)]
     mean = mean_fits(file_list)
     plot(mean)
     locations = []
     for file in file_list:
          brightest = brightest_pixel(image)
          locations.append(brightest)