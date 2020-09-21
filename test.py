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
    parent_directory = sys.argv[1]
else:
    # NOTE: Only need to change this in order to run test.py
    parent_directory = '/home/bacon/code/python_toys/'


# our list of fits file paths of our pulsar images
path = parent_directory + 'astronomy_data_analysis/sample_data/pulsar_images/'
file_list = [path + '000{0}.fits'.format(i) for i in range(0, 9)]
image_list = ada.fits_tools.get_data_stack(file_list)
ada.fits_tools.plot_images(image_list, title="Pulsar Images", ncols=3)


# demonstrating that the space efficient mean (scalable mean) and mean produce identical results
space_mean = ada.image_stacking.scalable_mean_fits(image_list)
mean = ada.image_stacking.mean_fits(image_list)
assert np.all(np.isclose(space_mean, mean))


# demonstrating the space efficient median and median
path = parent_directory + 'astronomy_data_analysis/sample_data/median_images/'
file_list = [path + 'image{0}.fits'.format(i) for i in range(0, 5)]
image_list = ada.fits_tools.get_data_stack(file_list)

median = ada.image_stacking.median_fits(image_list)
space_median = ada.image_stacking.scalable_median_fits(image_list, num_bins=100)
quit()

# demonstrating another function from image_stacking
locations = []
for image in image_list:
    brightest = ada.image_stacking.brightest_pixel(image)
    locations.append(brightest)
print("Locations of the brightest pixels: ", locations)


# some examples from naive_cross_matching
print("Starting crossmatching examples...")
path = parent_directory + 'astronomy_data_analysis/sample_data/star_catalogs/'
bss_cat = ada.cross_matching_tools.load_bss(path)
cosmos_cat = ada.cross_matching_tools.load_cosmos(path)


# First example
start = time.time()
max_dist = np.radians(5/3600)
matches, no_matches = ada.naive_cross_matching.naive_crossmatch(bss_cat, cosmos_cat, max_dist)
elapsed = time.time() - start
print("This is the time taken by naive crossmatch ", elapsed)
print("Number of matches with small search radius: ", len(matches))
print("Number of unmatched objects with small search radius: ", len(no_matches))


# Second example
max_dist = np.radians(40/3600)
matches, no_matches = ada.naive_cross_matching.naive_crossmatch(bss_cat, cosmos_cat, max_dist)
print("Number of matches with large search radius: ", len(matches))
print("Number of unmatched objects with large search radius: ", len(no_matches))


# First example, using k-d implementation
start = time.time()
kd_matches, kd_no_matches = ada.kd_cross_matching.crossmatch(bss_cat, cosmos_cat, max_dist=np.radians(max_dist))
elapsed = time.time() - start
print("This is the time taken by kd crossmatch ", elapsed)


# TODO
# Verify scipy kd tree stores same data as build_kd_tree()
scipy_tree = scipy.spatial.KDTree(bss_cat[:, :2])
print("this is the scipy kd tree ", scipy_tree)
