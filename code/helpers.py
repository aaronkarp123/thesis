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

# and IPython.display for audio output
import IPython.display as ipd

import glob
import os

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

import math

from features import *

def test_sound(filepath, segment_length = 0.2):
    # Load an audio file and analyze its content, returning the information necessary to query the file against the database
    
    try:
        y, sr = librosa.load(filepath)
    except Exception as e:
        print()
        print(e)
        print("Error loading file.")
        print()
        return

    # distance between end of file and multiple of segment_length
    segment_length_samples = segment_length * sr
    to_pad = segment_length_samples - y.size % segment_length_samples
    if(to_pad == segment_length_samples): # if you're on-the-money don't do anything
        to_pad = 0

    # pad to the correct length (a multiple of segment_length)
    y = np.pad(y, (0,int(to_pad)),'constant', constant_values=(0))
    num_segs = int(y.size / segment_length_samples)

    # add empty list to store file's feature data
    spectrograms = []
    mfccs = []
    rmss = []
    centroids = []

    for i in range((num_segs)):
        start = int(i * segment_length_samples)
        end = int((i+1) * segment_length_samples)
        features = get_all(y[start:end], sr)  # calculate features for sub-segment of signal
        spectrograms.append(features[0])
        mfccs.append(features[1])
        rmss.append(features[2])
        centroids.append(features[3])

    spectrogram_by_seg, spectrogram_by_seg_flat = segment_spectrograms([spectrograms], num_segs, 1)
    return y, sr, spectrogram_by_seg_flat


def query_sound(filename, engines, num_files, sounds=None, samplerates=None, display=False):
    # Take a single analyzed sound and return a list of ANN from a set of hashed databases (engines)
    
    y, sr, spec = test_sound(filename)
    
    scores = [0]*num_files
    distances = [0]*num_files
    cur_seg = 0
    for engine in engines:
        if (cur_seg >= len(spec)):
            break
        # Get nearest neighbours
        N = engine.neighbours(spec[cur_seg][0])
        for entry in N:
            index = parse_index(entry[1])
            if not math.isnan(entry[2]):
                scores[index] += 1
                distances[index] += entry[2]
        cur_seg += 1

    for i in range(len(scores)):
        if scores[i] == 0:
            distances[i] = 999999990

    guesses = sorted(range(len(distances)), key=lambda k : distances[k])
        
    if display:
        if sounds == None or samplerates == None:
            print("Must include sounds and samplerates for display")
            return guesses, distances
        best_guess = guesses[0]
        second_best_guess = guesses[1]

        print("Original query: " + filename)
        ipd.display(ipd.Audio(y, rate = sr)) # load search file
        print("Best guess: " + str(best_guess))
        ipd.display(ipd.Audio(sounds[best_guess], rate = samplerates[best_guess])) # load matched file
        print("Second best guess: " + str(second_best_guess))
        ipd.display(ipd.Audio(sounds[second_best_guess], rate = samplerates[second_best_guess])) # load matched file

        plt.figure(figsize=(12,4))
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.title('mel power spectrogram for Original Query')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()

        plt.figure(figsize=(12,4))
        S = librosa.feature.melspectrogram(sounds[best_guess], sr=samplerates[best_guess], n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(log_S, sr=samplerates[best_guess], x_axis='time', y_axis='mel')
        plt.title('mel power spectrogram for Best Guess: ' + str(best_guess))
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()


        plt.figure(figsize=(12,4))
        S = librosa.feature.melspectrogram(sounds[second_best_guess], sr=samplerates[second_best_guess], n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(log_S, sr=samplerates[second_best_guess], x_axis='time', y_axis='mel')
        plt.title('mel power spectrogram for Second Best Guess: ' + str(second_best_guess))
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        
        plt.show()
    return guesses, distances
    

def load_sounds(filedir= '../testSounds'):
    # Return all sounds within the directory as a list of numpy arrays

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
    # Analyze all sounds in (audiodata) for various audio features
    
    spectrograms = []
    mfccs = []
    rmss = []
    centroids = []
    max_num_segs = 0
    cur_file = 0
    avg_num_segs = 0
    cur_percentage = 0

    print("Analyzing...")

    for sound in audiodata:
        y = sound
        sr = rates[cur_file]
        
        if (round(cur_file / len(audiodata) * 100) != cur_percentage):
            cur_percentage = round(cur_file / len(audiodata) * 100)
            print(str(cur_percentage) + "%     ", end='')

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
    # Save numpy data in .npy files
    
    for file in data:
        name = file[0]
        np.save(directory + name + '.npy', file[1])
        print('saved ' + name)

def vectorize(ar):
    # Stack matrix into one long horizontal vector by time-slice (columns -> rows)
    return ar.flatten('F')

def build_lsh(data, hashbits=10):
    # Build a locality sensitive hashed database with (data), of bit-depth (hashbits)
    
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

def segment_spectrograms(specs, max_segs, file_count):
    # Re-order spectrograms by segment
    spect_by_seg = []
    spect_by_seg_flat = []
    for i in range(max_segs):
        seg = np.empty((file_count,128,specs[0][0].shape[1]))
        seg[:] = np.nan
        seg_flat = np.empty((file_count,128*specs[0][0].shape[1]))
        seg_flat[:] = np.nan
        for j in range(file_count):
            if i < len(specs[j]):
                seg[j] = specs[j][i]
                seg_flat[j] = vectorize(specs[j][i])
        spect_by_seg.append(seg)
        spect_by_seg_flat.append(seg_flat)
    return spect_by_seg, spect_by_seg_flat