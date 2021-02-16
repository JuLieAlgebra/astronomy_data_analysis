"""
stump for implementing and benchmarking sklearn's ball tree class with Haversine distance metric
"""
#from . import cross_matching_tools as cm_tools
import sklearn.neighbors
import numpy as np
import os

####################### Borrowed from cm_tools while in Dev. ################################

def hms2deg(hours, minutes, seconds):
    """
    Takes in Right Ascension (astronomical coordinate) in hour, minute, second
    format and and converts to degrees.

    Parameters
    ----------
    Hours:   int
    Minutes: int
    Seconds: float
    """
    return 15 * (hours + minutes / 60.0 + seconds / (60.0**2))


def dms2deg(degrees, minutes, seconds):
    """
    Takes in Declination (astronomical coordinate) in degree, minute, second
    format and and converts to degrees.

    Parameters
    ----------
    Degrees: int
    Minutes: int
    Seconds: float
    """
    return np.sign(degrees) * (abs(degrees) + minutes / 60.0 + seconds / (60.0**2))


def load_bss(path):
    """
    Reads in .dat file of the BSS catalog and returns numpy array of filtered data
    Data is a 2D np array of each object's location in RA, DEC.
    Sample row is [RA: radians, DEC: radians]
    Data has shape (number of entries in catalog, 2)

    Parameters
    ----------
    path: string to where bss catalog is stored
    """
    raw_data = np.genfromtxt(path+'bss.dat', usecols=range(1, 7))
    data_shape = (raw_data.shape[0], 2)
    data = np.zeros(data_shape)
    for ID, line in enumerate(raw_data):
        row = np.zeros(2)
        row[0] = np.radians(hms2deg(line[0], line[1], line[2]))
        row[1] = np.radians(dms2deg(line[3], line[4], line[5]))
        data[ID] = row
    return data


def load_cosmos(path):
    """
    Reads in .csv file of the Super COSMOS catalog and returns numpy array of filtered data
    Data is a 2D np array of each object's location in RA, DEC.
    Sample row is [RA: radians, DEC: radians]
    Data has shape (number of entries in catalog, 2)

    Parameters
    ----------
    path: string to where bss catalog is stored
    """
    raw_data = np.genfromtxt(path+'superCOSMOS.csv', delimiter=',', skip_header=1, usecols=[0, 1])
    data = np.radians(raw_data)
    data = data.reshape((raw_data.shape[0], raw_data.shape[1]))

    return data


##################################################################


def build_ball_tree(data, leaf_size=2):
    """
    """
    dist = sklearn.neighbors.DistanceMetric.get_metric('haversine')
    tree = sklearn.neighbors.BallTree(data, leaf_size=leaf_size, metric=dist)

    return tree


def cross_match(catalog1, catalog2, tolerance=np.radians(5/3600)):
    """
    Finds all of the matches of catalog1 in catalog2 with ball tree nearest neighbor search by assuming that entries that are
    close spatially are the same object in both catalogs.

    ball tree is constructed from catalog2. If catalog2 is smaller than catalog1, in order to avoid over-matching,
    that convention is swapped.

    no_matches includes the ID of the star without a match, the closest object in catalog 2, and the distance
    between them. Closest object found by the nearest neighbor search, which was not within tolerance.

    matches includes the IDs of the two stars from each catalog and the distance between
    them.

    Parameters
    ----------
    catalog 1, 2: 2D numpy arrays. See cross_matching_tools.load_bss, load_cosmos
    tolerance:    float, tolerance for matching stars in radians


    Returns
    -------
    no_matches: 2D numpy array, each entry [cat1 ID: int, cat2 ID: int, distance between the two: float]
    matches:    2D numpy array, each entry [cat1 ID: int, closest cat2 ID: int, distance between the two: float]
    """
    if len(catalog1) >= len(catalog2):
        print("WARNING: catalog1 is larger than catalog2. Switching to searching catalog1 for NN's of catalog2.")
        tree_catalog = catalog1
        NN_catalog = catalog2
    else:
        tree_catalog = catalog2
        NN_catalog = catalog1

    tree = build_ball_tree(tree_catalog)
    initial_results = tree.query_radius(NN_catalog, r=tolerance, return_distance=True, sort_results=True)

    """
    # In the middle of separating out matches and no matches from intial_results
    matches = []
    no_matches = []
    for dist, match in initial_results:
        print(dist, match)
        if dist <= tolerance:
            matches.append((match, dist))
        else:
            no_matches.append((match, dist))

    return np.array(matches), np.array(no_matches)
    """
    return initial_results

# hard coded path for initial sanity checking
path = "/home/bacon/code/python_toys/astronomy_data_analysis/sample_data/star_catalogs/"
bss_cat = load_bss(path)
cosmos_cat = load_cosmos(path)
matches = cross_match(bss_cat, cosmos_cat, tolerance=np.radians(5/3600))
print(matches)