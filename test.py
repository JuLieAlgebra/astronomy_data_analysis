#!/usr/bin/env python3
"""
Proper mess, but some examples.
TODO: Clean up
"""
import sys # to read in command line arguments
import astronomy_data_analysis as ada

import scipy.spatial
import numpy as np
import time

# user can provide the path to fits files via the command line or by changing the variable path below
path = '/home/bacon/code/datasets/astronomy_data/pulsar images/'
if len(sys.argv) > 1:
     fits_file = sys.argv[1]
else:
     fits_file = path + '0000.fits'


# some examples from image_stacking
image = ada.image_stacking.get_fits_data(fits_file)
ada.image_stacking.plot(image)


file_list = [path + '000{0}.fits'.format(i) for i in range(0, 9)]
mean = ada.image_stacking.mean_fits(file_list)
ada.image_stacking.plot(mean)
locations = []
for file in file_list:
     brightest = ada.image_stacking.brightest_pixel(image)
     locations.append(brightest)


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