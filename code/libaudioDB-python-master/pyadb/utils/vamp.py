# libaudioDB-python pyadb.utils.vamp
#
# This module provides VAMP utilities for use with pyadb.
#
# Copyright (C) 2015 Richard Lewis, Goldsmiths College
# Author: Richard Lewis <richard.lewis@gold.ac.uk>

import numpy as np
import csv

def _hastimes(filename=None, timescol=0, maxrows=50, threshold=0.1):
    """Attempts to determine if the given CSV feature file includes
    times. Returns bool.

    """
    with open(filename, 'rb') as csvfile:
        vampcsv = csv.reader(csvfile)
        deltas = []
        prev = 0.0
        i = 0
        for row in vampcsv:
            deltas.append(float(row[timescol]) - prev)
            prev = float(row[timescol])
            i += 1
            if i > maxrows:
                break

    return max(deltas) - min(deltas) < threshold

def _dims(filename=None, hastimes=None):
    """Determines the number of dimensions of the feature in the given
    feature file. If the hastimes argument is None, it uses _hastimes
    to determine whether the feature file contains times. Otherwise,
    hastimes should be a bool. Returns int.

    """
    with open(filename, 'rb') as csvfile:
        vampcsv = csv.reader(csvfile)
        cols = len(vampcsv.next())

    if hastimes or _hastimes(filename):
        return cols - 1
    else:
        return cols

def _load(filename=None, withtimes=False, hastimes=None):
    """Load the given feature CSV file. If withtimes is True, the times
    from the feature file will be returned too. If the hastimes
    argument is None, it uses _hastimes to determine whether the
    feature file contains times. Otherwise, hastimes should be a
    bool. Returns tuple: (array [times] or None, array [features]).

    """
    # determine the colums that make up the feature data
    if hastimes is None:
        hastimes = _hastimes(filename)

    dims = _dims(filename, hastimes=hastimes)
    if hastimes:
        fstcol = 1
        lstcol = dims + 1
    else:
        fstcol = 0
        lstcol = dims

    if fstcol == lstcol:
        cols = (fstcol,)
    else:
        cols = tuple(range(fstcol, lstcol))

    # load the features from the CSV file
    features = np.genfromtxt(filename, delimiter=',', usecols=cols)
    if withtimes:
        times = np.genfromtxt(filename, delimiter=',', usecols=(0,)).flatten()
    else:
        times = None

    return (times, features)

def features(featuresfile=None, powersfile=None, withtimes=False, fhastimes=None, phastimes=None):
    """Loads the features from the given featuresfile CSV file. Optionally
    also specify a powersfile of CSV power features. If withtimes is
    True, the times from the features file will be returned too. If
    the (f|p)hastimes arguments are None, it uses _hastimes to
    determine whether the corresponding feature or power features
    files contains times. Otherwise, (f|p)hastimes should be a
    bools. Returns tuple: (array [times] or None, array [features],
    array [powers] or None).

    """

    (times, features) = _load(filename=featuresfile, withtimes=withtimes, hastimes=fhastimes)
    if powersfile:
        (_, powers) = _load(filename=powersfile, withtimes=False, hastimes=phastimes)
    else:
        powers = None

    return (times, features, powers)
