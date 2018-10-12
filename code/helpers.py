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
import shutil

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

import pickle
import math
import re

from features import *

def save_engines(engines, name, directory='/Users/aaronkarp/Documents/Thesis/Code/savedBases/'):
    # Save the list of engines in the directory
    seg_num = 0
    for engine in engines:
        to_save = directory+name+'_'+str(seg_num)+'.p'
        pickle.dump(engine,open(to_save,'wb'))
        seg_num += 1
        
def load_engines(name, directory='/Users/aaronkarp/Documents/Thesis/Code/savedBases/'):
    # Return a list of engines loaded from the directory containing the string name
    os.chdir(directory)
    files = []
    files.extend(glob.glob("*"+name+"*.p"))
    files.sort(key=natural_keys)
    engines = []
    for file in files:
        engines.append(pickle.load(open(file,'rb')))
    return engines

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def test_sound(filepath, segment_length = 0.2, match_type='spectrogram'):
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
    
    if match_type == 'spectrogram':
        spectrogram_by_seg, spectrogram_by_seg_flat = segment_matrix([spectrograms], num_segs, 1)
        return y, sr, spectrogram_by_seg_flat
    if match_type == 'mfcc':
        mfcc_by_seg, mfcc_by_seg_flat = segment_matrix([mfccs], num_segs, 1)
        return y, sr, mfcc_by_seg_flat
    if match_type == 'centroid':
        centroid_by_seg, centroid_by_seg_flat = segment_matrix([centroids], num_segs, 1)
        return y, sr, centroid_by_seg_flat
    return


def query_sound(filename, engines, num_files, sounds=None, samplerates=None, display=False, segment_length=0.2, match_type='spectrogram'):
    # Take a single analyzed sound and return a list of ANN from a set of hashed databases (engines)
    
    y, sr, mat = test_sound(filename, segment_length=segment_length, match_type=match_type)
    
    scores = [0]*num_files
    distances = [0]*num_files
    cur_seg = 0
    for engine in engines:
        if (cur_seg >= len(mat)):
            break
        # Get nearest neighbours
        N = engine.neighbours(mat[cur_seg][0])
        for entry in N:
            index = int(entry[1]) #parse_index(entry[1])
            if not math.isnan(entry[2]):
                scores[index] += 1
                distances[index] += entry[2]
        cur_seg += 1

    for i in range(len(scores)):
        if scores[i] == 0:
            distances[i] = 9

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
        if distances[second_best_guess] < 9:
            print("Second best guess: " + str(second_best_guess))
            ipd.display(ipd.Audio(sounds[second_best_guess], rate = samplerates[second_best_guess])) # load matched file
        
        if match_type == 'spectrogram':
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

            if distances[second_best_guess] < 9:
                plt.figure(figsize=(12,4))
                S = librosa.feature.melspectrogram(sounds[second_best_guess], sr=samplerates[second_best_guess], n_mels=128)
                log_S = librosa.power_to_db(S, ref=np.max)
                librosa.display.specshow(log_S, sr=samplerates[second_best_guess], x_axis='time', y_axis='mel')
                plt.title('mel power spectrogram for Second Best Guess: ' + str(second_best_guess))
                plt.colorbar(format='%+02.0f dB')
                plt.tight_layout()
            
        if match_type == 'mfcc':
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
            plt.figure(figsize=(12, 6))
            plt.subplot(3,1,1)
            librosa.display.specshow(mfcc, x_axis='time')
            plt.ylabel('MFCC for Original Query')
            plt.colorbar()
            plt.tight_layout()
            
            S = librosa.feature.melspectrogram(sounds[best_guess], sr=samplerates[best_guess], n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
            plt.figure(figsize=(12, 6))
            plt.subplot(3,1,1)
            librosa.display.specshow(mfcc, x_axis='time')
            plt.ylabel('MFCC for Best Guess: ' + str(best_guess))
            plt.colorbar()
            plt.tight_layout()
            
            if distances[second_best_guess] < 9:
                S = librosa.feature.melspectrogram(sounds[second_best_guess], sr=samplerates[second_best_guess], n_mels=128)
                log_S = librosa.power_to_db(S, ref=np.max)
                mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
                plt.figure(figsize=(12, 6))
                plt.subplot(3,1,1)
                librosa.display.specshow(mfcc, x_axis='time')
                plt.ylabel('MFCC for Second Best Guess: ' + str(second_best_guess))
                plt.colorbar()
                plt.tight_layout()
            
        if match_type == 'centroid':
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.semilogy(cent.T, label='Spectral centroid of Original Query')
            plt.ylabel('Hz')
            plt.xticks([])
            plt.xlim([0, cent.shape[-1]])
            plt.legend()
            
            cent = librosa.feature.spectral_centroid(sounds[best_guess], sr=samplerates[best_guess])
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.semilogy(cent.T, label='Spectral centroid of Best Guess: ' + str(best_guess))
            plt.ylabel('Hz')
            plt.xticks([])
            plt.xlim([0, cent.shape[-1]])
            plt.legend()
            
            if distances[second_best_guess] < 9:
                cent = librosa.feature.spectral_centroid(sounds[second_best_guess], sr=samplerates[second_best_guess])
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.semilogy(cent.T, label='Spectral centroid of Second Best Guess: ' + str(second_best_guess))
                plt.ylabel('Hz')
                plt.xticks([])
                plt.xlim([0, cent.shape[-1]])
                plt.legend()
            
        if match_type == 'rms':
            S, phase = librosa.magphase(librosa.stft(y))
            rms = librosa.feature.rmse(S=S)
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.semilogy(rms.T, label='RMS Energy for Original Query')
            plt.xticks([])
            plt.xlim([0, rms.shape[-1]])
            plt.legend(loc='best')
            
            S, phase = librosa.magphase(librosa.stft(sounds[best_guess]))
            rms = librosa.feature.rmse(S=S)
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.semilogy(rms.T, label='RMS Energy for Best Guess: ' + str(best_guess))
            plt.xticks([])
            plt.xlim([0, rms.shape[-1]])
            plt.legend(loc='best')
            
            if distances[second_best_guess] < 9:
                S, phase = librosa.magphase(librosa.stft(sounds[second_best_guess]))
                rms = librosa.feature.rmse(S=S)
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.semilogy(rms.T, label='RMS Energy for Second Best Guess: ' + str(second_best_guess))
                plt.xticks([])
                plt.xlim([0, rms.shape[-1]])
                plt.legend(loc='best')
        
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
        engine.store_vector(data[index], '%d' % index)

    return engine


def parse_index(index):
    s1,s2 = index.split('_')
    return int(s2)

def segment_matrix(mat, max_segs, file_count):
    # Re-order spectrograms by segment
    mat_by_seg = []
    mat_by_seg_flat = []
    for i in range(max_segs):
        seg = np.empty((file_count,mat[0][0].shape[0],mat[0][0].shape[1]))
        seg[:] = np.nan
        seg_flat = np.empty((file_count,mat[0][0].shape[0]*mat[0][0].shape[1]))
        seg_flat[:] = np.nan
        for j in range(file_count):
            if i < len(mat[j]):
                seg[j] = mat[j][i]
                seg_flat[j] = vectorize(mat[j][i])
        mat_by_seg.append(seg)
        mat_by_seg_flat.append(seg_flat)
    return mat_by_seg, mat_by_seg_flat

    
def reboot_directory(path='/Users/aaronkarp/Documents/Thesis/Code/savedBases'):
    # Remove and remake target directory
    shutil.rmtree(path)
    os.mkdir(path)
