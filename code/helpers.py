# We'll need numpy for some mathematical operations
import numpy as np

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

import glob
import os

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

from features import *

def load_sounds(filedir= '../testSounds'):

    os.chdir(filedir)
    audiodata = []
    rates = []
    files = []
    files.extend(glob.glob("*.wav"))
    files.extend(glob.glob("*.flac"))
    cur_file = 0
    cur_percentage = 0

    print("Loading...")

    for file in files:
        # print(file + "   ", end='')
        if (round(cur_file / len(files) * 100) != cur_percentage):
            cur_percentage = round(cur_file / len(files) * 100)
            print(str(cur_percentage) + "%     ", end='')
        
        try:
            y, sr = librosa.load(filedir + "/" + file)
        except Exception as e:
            print()
            print(e)
            print("Error loading file. Continuing to next file.")
            print()
            continue
        audiodata.append(y)
        rates.append(sr)
        cur_file += 1

    print("Finished Loading")
    return audiodata, rates

def analyze_sounds(audiodata, rates, segment_length = 0.2):
    spectrograms = []
    mfccs = []
    rmss = []
    centroids = []
    max_num_segs = 0
    cur_file = 0
    avg_num_segs = 0

    print("Analyzing...")

    for sound in audiodata:
        y = sound
        sr = rates[cur_file]

        # distance between end of file and multiple of segment_length
        segment_length_samples = segment_length * sr
        to_pad = segment_length_samples - y.size % segment_length_samples
        if(to_pad == segment_length_samples): # if you're on-the-money don't do anything
            to_pad = 0
        
        # pad to the correct length (a multiple of segment_length)
        #print(str(y.size/sr) + " seconds + " + str(to_pad/sr) + " seconds = ", end='')
        y = np.pad(y, (0,int(to_pad)),'constant', constant_values=(0))
        #print(str(y.size/sr) + " seconds")
        num_segs = int(y.size / segment_length_samples)
        
        avg_num_segs += num_segs

        # add empty list to store file's feature data
        spectrograms.append([])
        mfccs.append([])
        rmss.append([])
        centroids.append([])
        
        for i in range((num_segs)):
            start = int(i * segment_length_samples)
            end = int((i+1) * segment_length_samples)
            features = get_all(y[start:end], sr)  # calculate features for sub-segment of signal
            rates.append(sr)
            spectrograms[cur_file].append(features[0])
            mfccs[cur_file].append(features[1])
            rmss[cur_file].append(features[2])
            centroids[cur_file].append(features[3])
        
        if num_segs > max_num_segs:
            max_num_segs = num_segs
            
        cur_file += 1
    print("Finished Analyzing")
    print("\nAverage number of segments: ", str(avg_num_segs/cur_file))

    return spectrograms, mfccs, rmss, centroids, max_num_segs

def save_data(data, directory='/Users/aaronkarp/Documents/Thesis/Code/savedData/'):
    for file in data:
        name = file[0]
        np.save(directory + name + '.npy', file[1])
        print('saved ' + name)


# Stack matrix into one long horizontal vector by time-slice (columns -> rows)
def vectorize(ar):
    return ar.flatten('F')

def build_lsh(data, hashbits=10):
    dimensions = data.shape[1]

    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', hashbits)

    # Create engine with pipeline configuration
    engine = Engine(dimensions, lshashes=[rbp])

    # Index 1000000 random vectors (set their data to a unique string)
    for index in range(len(data)):
        engine.store_vector(data[index], 'data_%d' % index)

    return engine


def parse_index(index):
    s1,s2 = index.split('_')
    return int(s2)