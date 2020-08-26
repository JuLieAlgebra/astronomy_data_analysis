"""
See naive_cross_matching first for context and other library functions used here.

Implements a nearest neighbor search using k-d trees to partition the space.
K-d trees work by dividing the data into leafs of a binary tree according to their RA and DEC values
and work in O(n * log(n)), a lot nicer than O(n**2)!
"""
from . import cross_matching_tools as cm_tools
import numpy as np


def build_kd_tree(catalog, depth=0):
    """
    Recursive algorithm to build a k-d tree from catalog
    K-d tree is stored in a nested dictionary.

    Parameters
    ----------
    catalog: 2D numpy array. Ex: catalog[i] = [RA, DEC, ID]. See cross_matching_tools.load_bss, load_cosmos
    depth: int, for recursion.
    """
    n = len(catalog)

    if n <= 0:
        return None

    coord = depth % 2

    sorted_data = catalog[catalog[:,coord].argsort()] # will sort numpy array by column instead of by axis

    return {
        'star': sorted_data[n // 2],
        'left': build_kd_tree(sorted_data[:n // 2], depth + 1),
        'right': build_kd_tree(sorted_data[n // 2 + 1:], depth + 1)
    }


def closer_star(root, s1, s2):
    """
    Helper function for find_closest_star.
    Given three k-d tree nodes, returns closest star to root.

    Parameters
    ----------
    Root: element of kd tree
    s1: element of kd tree
    s1: element of kd tree

    """
    if s1 is None:
        return s2

    if s2 is None:
        return s1

    d1 = cm_tools.angular_dist(s1[0], s1[1], root[0], root[1])
    d2 = cm_tools.angular_dist(s2[0], s2[1], root[0], root[1])

    if d1 < d2:
        return s1
    else:
        return s2


def find_closest_star(root, star, depth=0):
    """
    For one star, recursively finds closest match in entire k-d tree-ized catalog.
    Returns element of kd tree

    Parameters
    ----------
    Root: element of kd tree
    Star: element of kd tree
    """
    if root is None:
        return None

    coord = depth % 2

    next_branch = None
    opposite_branch = None

    if star[coord] < root['star'][coord]:
        next_branch = root['left']
        opposite_branch = root['right']

    else:
        next_branch = root['right']
        opposite_branch = root['left']

    best = closer_star(star, find_closest_star(next_branch, star, depth + 1), root['star'])

    best_distance = cm_tools.angular_dist(star[0], star[1], best[0], best[1])

    if best_distance > (star[coord] - root['star'][coord]) ** 2:
        best = closer_star(star, find_closest_star(opposite_branch, star, depth + 1), best)

    return best


def crossmatch(catalog1, catalog2, max_dist):
    """
    Finds all of the matches of catalog1 in catalog2 by assuming that entries that are
    close spatially are the same object in both catalogs.

    no_matches includes the ID of the star without a match and the closet star found
    by the nearest neighbor search, which was not within max_dist.

    matches includes the IDs of the two stars from each catalog and the distance between
    them.

    Parameters
    ----------
    catalog 1, 2: 2D numpy arrays. See cross_matching_tools.load_bss, load_cosmos
    max_dist: float, tolerance for matching stars in radians
    """
    tree = build_kd_tree(catalog2)
    matches = []
    no_matches = []

    for star in catalog1:
        match = find_closest_star(tree, star)
        dist = cm_tools.angular_dist(match[0], match[1], star[0], star[1])
        if dist < max_dist:
            matches.append((star[2], match[2], dist))
        else:
            no_matches.append((star[2], match[2]))

    return matches, no_matches
