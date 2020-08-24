"""
In astronomy, we often care about what an object looks like in all bands of the EM
spectrum - radio, observable, etc. The process of matching up one object with all of
the observations of that object across different catalogs is called cross-matching.

This is a naive O(n**2) cross matching algorithm implemented from scratch.
"""
from . import cross_matching_tools as cm_tools
import numpy as np


def find_closest(data, ra, dec):
     """
     Takes in filtered data of one catalog of objects and one set of coordinates
     for the target. Finds the closest object to the set of coordinates in the first catalog.

     Returns tuple of the ID (index of star in catalog) and angular distance between
     target and the closest object to the target.
     """
     best_match = -1
     closest_dist = np.inf

     for ID, star in enumerate(data):
          dist = cm_tools.angular_dist(star[0], star[1], ra, dec)
          if dist < closest_dist:
               best_match = ID
               closest_dist = dist

     return best_match, closest_dist


def naive_crossmatch(bss, cosmos, max_dist):
     """
     Given two catalogs of objects, the bss and super cosmos, will match up the objects with
     objects in the other catalog.

     The idea is to find the observations that correspond to the same object in each catalog
     and pair them with each other.

     If there is no closest object for a star within max_dist, we assume that there isn't
     another observation of the object in the second catalog, so we add it to a list of no matches
     and return it too.

     Returns tuple of matches (catalog 1 ID, catalog 2 ID, distance between them) and no matches
     (catalog 1 ID).

     NOTE: catalog ID refers to the index of an object in a catalog (catalog[ID] = object)
     """
     matches = []
     no_matches = []
     for bID, bstar in enumerate(bss):
          ID, dist = find_closest(cosmos, bstar[0], bstar[1])
          if dist < max_dist:
               matches.append((bID, ID, dist))  # this matches the bss star ID to the closest cosmos star ID
          else:
               no_matches.append(bID)

     return matches, no_matches
