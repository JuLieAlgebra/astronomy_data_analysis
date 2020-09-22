#!/usr/bin/env python3
"""
Proper mess, but some examples.
TODO: Switch to using ipython notebook instead
"""
import time
import os
import scipy.spatial
import numpy as np
import astronomy_data_analysis as ada

# Establishing working directory
parent_directory = os.path.dirname(__file__)


# Our list of fits file paths of our pulsar images
path = os.path.join(parent_directory, 'sample_data', 'pulsar_images', '')
file_list = [path + '000{0}.fits'.format(i) for i in range(0, 9)]
pulsar_list = ada.fits_tools.get_data_stack(file_list)
ada.fits_tools.plot_images(pulsar_list, title="Pulsar Images", ncols=3, subtitles=['Image {0}'.format(i) for i in range(0, 9)])


# Demonstrating that the scalable efficient mean (scalable mean) and mean produce identical results
path = os.path.join(parent_directory, 'sample_data', 'median_images', '')
file_list = [path + 'image{0}.fits'.format(i) for i in range(0, 5)]
median_images = ada.fits_tools.get_data_stack(file_list)

scalable_mean, scalable_std = ada.image_stacking.scalable_stats_fits(file_list)
mean = ada.image_stacking.mean_fits(median_images)
std = ada.image_stacking.std_fits(median_images)
assert np.all(np.isclose(scalable_mean, mean))
print("Passed: Space efficient mean test")
assert np.all(np.isclose(scalable_std, std))
print("Passed: Space efficient standard deviation test")
print('===================================')


# Demonstrating the scalable efficient median and median
median = ada.image_stacking.median_fits(median_images)
scalable_median = ada.image_stacking.scalable_median_fits(file_list, num_bins=100)
ada.fits_tools.plot_images( np.array([median, scalable_median]),
                            title="Median Images", ncols=2,
                            subtitles=['Numpy Median', 'Binapprox Median'])


# Demonstrating another function from image_stacking
locations = []
for image in pulsar_list:
    brightest = ada.image_stacking.brightest_pixel(image)
    locations.append(brightest)
print("Locations of the brightest pixels: ", locations)
print('===================================')


# Some examples from naive_cross_matching
print("Starting crossmatching examples...")
path = os.path.join(parent_directory, 'sample_data', 'star_catalogs', '')
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
print('===================================')


# Second example
max_dist = np.radians(40/3600)
matches, no_matches = ada.naive_cross_matching.naive_crossmatch(bss_cat, cosmos_cat, max_dist)
print("Number of matches with large search radius: ", len(matches))
print("Number of unmatched objects with large search radius: ", len(no_matches))
print('===================================')


# First example, using k-d implementation
start = time.time()
kd_matches, kd_no_matches = ada.kd_cross_matching.crossmatch(bss_cat, cosmos_cat, max_dist=np.radians(max_dist))
kd_elapsed = time.time() - start
print("This is the time taken by k-d crossmatch ", kd_elapsed)
print("This is the ratio of naive crossmatch runtime to k-d runtime: ", round(elapsed / kd_elapsed, 2))
print('===================================')


# TODO
# Verify scipy kd tree stores same data as build_kd_tree()
scipy_tree = scipy.spatial.KDTree(bss_cat[:, :2])
print("This is the scipy kd tree ", scipy_tree)
