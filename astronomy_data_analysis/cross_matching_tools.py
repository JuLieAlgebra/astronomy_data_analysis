"""
Functions for loading and handling star catalog data. Includes distance metric
for star objects.
"""
import numpy as np
import os.path

##################################################################################

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


##################################################################################

def load_bss(path):
    """
    Reads in .dat file of the BSS catalog and returns numpy array of filtered data
    Data is a 2D np array of each object's location in RA, DEC
    Sample row is [RA: radians, DEC: radians, ID: float]
    Data has shape (number of entries in catalog, 3)

    Parameters
    ----------
    path: string to where bss catalog is stored
    """
    if os.path.isfile(path+'bss_catalog.npy'):
        data = np.load(path+'bss_catalog.npy')

    else:
        raw_data = np.genfromtxt(path+'bss.dat', usecols=range(1, 7))
        data_shape = (raw_data.shape[0], 3)
        data = np.zeros(data_shape)

        for ID, line in enumerate(raw_data):
            row = np.zeros(3)
            row[0] = np.radians(hms2deg(line[0], line[1], line[2]))
            row[1] = np.radians(dms2deg(line[3], line[4], line[5]))
            row[2] = ID
            data[ID] = row

        np.save(path+"bss_catalog", data)

    return data


def load_cosmos(path):
    """
    Reads in .csv file of the Super COSMOS catalog and returns numpy array of filtered data
    Data is a 2D np array of each object's location in RA, DEC and its ID.
    Sample row is [RA: radians, DEC: radians, ID: float]
    Data has shape (number of entries in catalog, 3)

    Parameters
    ----------
    path: string to where bss catalog is stored
    """
    if os.path.isfile(path+'superCOSMOS_catalog.npy'):
        data = np.load(path+'superCOSMOS_catalog.npy')

    else:
        raw_data = np.genfromtxt(path+'superCOSMOS.csv', delimiter=',', skip_header=1, usecols=[0, 1])
        data = np.column_stack((np.radians(raw_data), np.arange(len(raw_data))))

        np.save(path+"superCOSMOS_catalog", data)

    return data

##################################################################################

def angular_dist(ra_1, dec_1, ra_2, dec_2):
    """
    Takes in RA and DEC of two objects in radians
    Computes Haversine Formula  ( https://en.wikipedia.org/wiki/Haversine_formula )
    for distances between objects ON celestial sphere
    Return angular distance in radians

    NOTE:   this is NOT the distance between objects in space, this is only the ANGULAR distance
            between objects ON the celestial sphere

    Parameters
    ----------
    ra_1:   float, radians
    dec_1:  float, radians
    ra_2:   float, radians
    dec_2:  float, radians
    """

    radicand = np.sin(np.abs(dec_1 - dec_2) / 2.0)**2 + np.cos(dec_1) \
             * np.cos(dec_2) * np.sin(np.abs(ra_1 - ra_2) / 2.0)**2

    return 2 * np.arcsin(np.sqrt(radicand))

