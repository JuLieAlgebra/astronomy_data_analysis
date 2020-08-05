#!/usr/bin/env python3
"""  In astronomy, we often care about what an object looks like in all bands of the EM
     spectrum - radio, observable, etc. The process of matching up one object with all of
     the observations of that object across different catalogs is called cross matching.

     This is a naive O(n^2) cross matching algorithm implemented from scratch.
"""
import numpy as np
import os.path

def hms2dec(hours, minutes, seconds):
     """  Takes in Right Ascension (astronomical coordinate) in hour, minute, second
          format and and converts to degrees.
          Hour: int
          Minute: int
          Second: float
     """
     return 15*(hours+minutes/60.0 + seconds/(60.0**2))


def sign(num):
     """  Just returns the sign of a number. Input must be real.
     """
     if num < 0:
          return -1
     return 1


def dms2dec(degrees, minutes, seconds):
     """  Takes in Declination (astronomical coordinate) in degree, minute, second
          format and and converts to degrees.
          Hour: int
          Minute: int
          Second: float
     """
     return sign(degrees) * (abs(degrees) + minutes/60.0 + seconds/(60.0**2))


def import_bss(path):
     """  Reads in .dat file of bss catalog ( http://cdsarc.u-strasbg.fr/viz-bin/Cat?J/MNRAS/384/775 in table2.dat)
          and returns numpy array of filtered data
          Data is a np array of tuples, with each object in the bss catalog with its own tuple and ID
          Tuple = (ID: int, RA: degrees, DEC: degrees)
     """
     if os.path.isfile(path+'bss_catalog.npy'):
          data = np.load(path + 'bss_catalog.npy')

     else:
          raw_data = np.genfromtxt(path+'bss.dat', usecols=range(0, 7))

          data_shape = (raw_data.shape[0], 3)
          data = np.zeros(data_shape[0], dtype='int32, float64, float64')

          for ID, line in enumerate(raw_data):
               tup = np.zeros(3)
               tup[0] = int(ID + 1)
               tup[1] = hms2dec(line[1], line[2], line[3])
               tup[2] = dms2dec(line[4], line[5], line[6])
               data[ID] = tuple(tup)

          np.save(path+"bss_catalog", data)

     return data


def import_super(path):
     """  Reads in .csv file of super COSMOS catalog ( http://ssa.roe.ac.uk/allSky in SCOS_XSC_mCl1_B21.5_R20_noStepWedges.csv.gz )
          and returns numpy array of filtered data
          Data is a np array of tuples, with each object in the bss catalog with its own tuple and ID
          Tuple = (ID: int, RA: degrees, DEC: degrees)
     """
     if os.path.isfile(path+'superCOSMOS_catalog.npy'):
          data = np.load(path + 'superCOSMOS_catalog.npy')

     else:
          raw_data = np.genfromtxt(path+'superCOSMOS.csv', delimiter=',', skip_header=1, usecols=[0, 1])
          data_shape = (raw_data.shape[0], 3)
          data = np.zeros(data_shape[0], dtype='int32, float64, float64')

          for ID, line in enumerate(raw_data):
               tup = np.zeros(3)
               tup[0] = int(ID + 1)
               tup[1] = line[0]
               tup[2] = line[1]
               data[ID] = tuple(tup)
          np.save(path+"superCOSMOS_catalog", data)

     return data


def angular_dist(ra_1, dec_1, ra_2, dec_2):
     """  Takes in RA and DEC of two objects in degrees
          Computes Haversine Formula  ( https://en.wikipedia.org/wiki/Haversine_formula )
          for distances between objects ON celestial sphere
          Return angular distance in degrees

          NOTE:     this is NOT the distance between objects in space, this is only the ANGULAR distance
                    between objects ON the celestial sphere
     """
     ra_1 = np.radians(ra_1)
     ra_2 = np.radians(ra_2)
     dec_1 = np.radians(dec_1)
     dec_2 = np.radians(dec_2)

     radicand = np.sin(np.abs(dec_1 - dec_2) / 2.0)**2 + np.cos(dec_1) \
                * np.cos(dec_2) * np.sin(np.abs(ra_1 - ra_2) / 2.0)**2

     return np.degrees(2 * np.arcsin( np.sqrt(radicand)))


def find_closest(data, ra, dec):
     """  Takes in filtered data of one catalog of objects and one set of coordinates
          for the target.
          Finds the closest object to the set of coordinates in the first catalog

          Returns tuple of the ID and angular distance between target and the closest
          object to the target.
     """
     best_match = -1
     closest_dist = np.inf

     for star in data:
          dist = angular_dist(star[1], star[2], ra, dec)
          if dist < closest_dist:
               best_match = star[0]
               closest_dist = dist

     return best_match, closest_dist


def crossmatch(bss, cosmos, max_dist):
     """  Given two catalogs of objects, the bss and super cosmos, will match up the objects with
          objects in the other catalog.

          The idea is to find the observations that correspond to the same object in each catalog and pair
          them with each other.

          If there is no closest object for a star within max_dist, then we add it to a list of no matches and
          return it too

          Returns tuple of matches (catalog 1 ID, catalog 2 ID, distance between them) and no matches (catalog 1 ID)
     """
     matches = []
     no_matches = []
     for bstar in bss:
          ID, dist = find_closest(cosmos, bstar[1], bstar[2])
          if dist < max_dist:
               matches.append((bstar[0], ID, dist))  # this matches the bss star ID to the closest cosmos star ID
          else:
               no_matches.append(bstar[0])

     return matches, no_matches


if __name__ == '__main__':

     path = '/home/bacon/code/datasets/astronomy_data/star_catalogs/'
     bss_cat = import_bss(path)
     super_cat = import_super(path)

     # First example in the question
     max_dist = 40/3600
     matches, no_matches = crossmatch(bss_cat, super_cat, max_dist)
     print(matches[:3])
     print(no_matches[:3])
     print(len(no_matches))

     # Second example in the question
     max_dist = 5/3600
     matches, no_matches = crossmatch(bss_cat, super_cat, max_dist)
     print(matches[:3])
     print(no_matches[:3])
     print(len(no_matches))
