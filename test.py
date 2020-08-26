#!/usr/bin/env python3
"""
Proper mess, but some examples.
TODO: Clean up
"""
import sys # to read in command line arguments
import scipy.spatial
import numpy as np
import time
import astronomy_data_analysis as ada

# user can provide the path to fits files via the command line or by changing the variable path below
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = '/home/bacon/code/datasets/astronomy_data/pulsar images/'


# our list of fits file paths of our pulsar images
file_list = [path + '000{0}.fits'.format(i) for i in range(0, 9)]
image_list = ada.fits_tools.get_data_stack(file_list)
ada.fits_tools.plot_images(image_list, title="Pulsar Images")


# demonstrating that the space efficient mean (scalable mean) and mean produce identical results
space_mean = ada.image_stacking.scalable_mean_fits(image_list)
mean = ada.image_stacking.mean_fits(image_list)
assert np.all(np.isclose(space_mean, mean))


# demonstrating another function from image_stacking
locations = []
for image in image_list:
    brightest = ada.image_stacking.brightest_pixel(image)
    locations.append(brightest)
print("Locations of the brightest pixels: ", locations)


# some examples from naive_cross_matching
path = '/home/bacon/code/datasets/astronomy_data/star_catalogs/'
bss_cat = ada.cross_matching_tools.load_bss(path)
cosmos_cat = ada.cross_matching_tools.load_cosmos(path)


# First example
max_dist = np.radians(40/3600)
matches, no_matches = ada.naive_cross_matching.naive_crossmatch(bss_cat, cosmos_cat, max_dist)
print(matches[:3])
print(no_matches[:3])


# Second example
start = time.time()
max_dist = np.radians(5/3600)
matches, no_matches = ada.naive_cross_matching.naive_crossmatch(bss_cat, cosmos_cat, max_dist)
elapsed = time.time() - start
print("this is the time taken by naive crossmatch ", elapsed)


# First example, using k-d implementation
start = time.time()
matches, no_matches = ada.kd_cross_matching.crossmatch(bss_cat, cosmos_cat, max_dist=np.radians(40/3600))
elapsed = time.time() - start
print(matches[:3])
print(no_matches[:3])
print("this is the time taken by kd crossmatch ", elapsed)


# TODO
# Verify scipy kd tree stores same data as build_kd_tree()
scipy_tree = scipy.spatial.KDTree(bss_cat[:, :2])
print("this is the scipy kd tree ", scipy_tree)