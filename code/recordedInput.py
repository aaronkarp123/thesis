from lstmHelpers import *
import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing import Process, Queue
from joblib import Parallel, delayed
import time
import sys
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import audioop
import argparse
from pythonosc import dispatcher, osc_server # https://pypi.org/project/python-osc/
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime

CHUNK = 1024  # CHUNKS of bytes to read each time from mic
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050

masking = True

RUNNING = Queue()
RUNNING.put(True)

audioq = Queue()
prespecQ = Queue()
output = Queue()
to_play = Queue()

processes = []

def flush():
    global audioq, prespecQ, output, to_play
    while not audioq.empty():
        audioq.get()
    while not prespecQ.empty():
        prespecQ.get()
    while not output.empty():
        output.get()
    while not to_play.empty():
        to_play.get()

def parse_index_file(path):
    filenames = []
    words = []
    print("Parsing index file")
    tree = ET.parse(path)
    root = tree.getroot()
    for child in root:
        if child.tag == 'group':
            root = child
            for child in root:
                filenames.append(child.attrib['path'])
                root2 = child
                for child2 in root2:
                    words.append(child2.attrib['swac_text'])
    print("Parsed index file: " + str(len(words)) + " files present")
    return filenames, words

def extract_class_info(matrix):
    data_dict = {}
    os.chdir('../../../../../../../../../Volumes/External Storage/Thesis/Corpus')
    print("Loading audio files...")
    for i in range(len(audio_matrix)):
        filename = audio_matrix[i][0]
        file_class = int(audio_matrix[i][1])
        if file_class not in data_dict:
            yt,sr = librosa.load(filename)
            y, idx = librosa.effects.trim(yt, top_db=40)
            data_dict[file_class] = y
    print("Loaded audio files\n")
    return data_dict

def output_calc(data_dict):
    global output, to_play, CHUNK

    cur_out = -1  # -2 = stop, -1 = silence
    new_temp_out = -1
    audio_file = 0
    start_pos = 0
    to_run = True
    last_n_outputs = []
    med_output = -1
    prev_run = time.time()
    while to_run:
        while not output.empty():
            new_temp_out = output.get()
            prev_run = time.time()
            if new_temp_out == -2:
                to_run = False
            if new_temp_out >= 0:
                last_n_outputs.append(new_temp_out)
                if len(last_n_outputs) > 5:
                    last_n_outputs.pop(0)
                med_output = max(set(last_n_outputs), key=last_n_outputs.count)
                print(med_output, end='  : ')
                print(last_n_outputs)
                new_temp_out = med_output
                if med_output != cur_out:
                    sys.stdout.flush()
                    start_pos = 0# int(22050/4)
                    audio_file = data_dict[med_output]
                    while not to_play.empty():
                        to_play.get()
        cur_out = new_temp_out
        if cur_out == -1:
            continue
        else:
            if time.time() - prev_run > 5:
                print("Flushing from output thread")
                flush()
                prev_run = time.time()
        if to_play.empty() and cur_out >= 0:
            if start_pos+CHUNK > len(audio_file):
                start_pos = 0
            data = audio_file[start_pos:start_pos+CHUNK]
            to_play.put(data)
            start_pos += CHUNK



def concur_process(model):
    global RUNNING, audioq, output

    audio_data = []
    time_window = int((2048*3) / CHUNK)

    to_run = True

    currently_above_threshold = False

    while to_run:
        while not audioq.empty():
            audio_data.append(audioq.get())
        if len(audio_data) >= time_window and prespecQ.empty():  # If you have a full time-window's worth of data
            audio_data = audio_data[-time_window:]  # Take the most recent time_window chunks
            audio_data_np = np.concatenate(np.array(audio_data)).ravel()
            rms = np.sqrt(np.mean(audio_data_np**2))
            if (rms > 0.02 and currently_above_threshold) or (rms > 0.03 and not currently_above_threshold):
                window = np.pad(audio_data_np, (0, 2048), 'constant', constant_values=(0.0,0.0))
                prespecQ.put(window)
                currently_above_threshold = True
            else:
                currently_above_threshold = False
                output.put(-1)
            print("Placed Audio Chunk. Above Threshold: " + str(currently_above_threshold))
        if not RUNNING.empty():
            to_run = RUNNING.get()

    print("Process Terminated")

def listen(p, output_stream, files, words, file_append):
    global RUNNING, audioq, output, to_play, CHUNK

    print("** Stream Opened")

    audio_string_chunk = ''

    num_loops = 300
    to_run = True

    scaler = MinMaxScaler(feature_range=(0, 1))

    f= open("timestamp_test_"+str(masking)+"2.txt","w+")
    
    files = files[num_loops:]

    last_prediction = time.time()

    for file in files:
        print(file_append + file)
        loaded_audio, sr = librosa.load(file_append + file)
        cur_frame = 0
        while cur_frame + CHUNK <= len(loaded_audio):
            try:
                audio_string_chunk = loaded_audio[cur_frame:cur_frame+CHUNK]
            except Exception as e:
                print("Dropped Chunk")
                cur_frame += CHUNK
                continue
            
            audioq.put(audio_string_chunk)

            if not prespecQ.empty():
                ad = prespecQ.get()
                spec = get_spectrogram(ad, 22050, n_mels=128, display=False)
                transposed = spec.T
                transposed = scaler.fit_transform(transposed)
                output.put(model.predict_classes(np.array([transposed[0:10]]), batch_size=1)[0])
                last_prediction = time.time()

            mod_output = audio_string_chunk
            if not to_play.empty() and masking:
                mod_output = np.add(mod_output, to_play.get())
            elif time.time() - last_prediction > 5:
                print("Flushing from Main Thread")
                flush()
                last_prediction = time.time()
            output_stream.write(mod_output.tobytes(), CHUNK)
            cur_frame += CHUNK

        now = datetime.now()
        f.write(str(datetime.timestamp(now)) + " : " + words[num_loops] + "\n")
        silence = np.zeros(22055*2)
        cur_frame = 0
        while cur_frame + CHUNK <= len(silence):
            output_stream.write(silence[cur_frame:cur_frame+CHUNK].tobytes(), CHUNK)
            cur_frame += CHUNK
        num_loops += 1

    RUNNING.put(False)
    to_run = False
    output.put(-2)

    print("** Stream Terminated")

    input_stream.close()
    output_stream.close()
    p.terminate()
    f.close()


if __name__ == '__main__':

    files, words = parse_index_file("../../../../../../../../../Volumes/External Storage/Thesis/Corpus/eng-balm-emmanuel/flac/index.xml")

    audio_matrix = load_stream(name="/trainingInfo_300.txt", filedir='lstmData')

    old_dir = os.getcwd()
    os.chdir('../../../../../../../../../Volumes/External Storage/Thesis/Corpus')
    data_dict = extract_class_info(audio_matrix)
    os.chdir(old_dir)

    model = load_model('lstmData/LSTMModel_300.h5')

    p = pyaudio.PyAudio()
    output_stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK)

    #lp = Process(target=listen, args=(p, stream,))
    #processes.append(lp)
    #lp.start()

    #concur_process(model)


    cp = Process(target=concur_process, args=(model,))
    processes.append(cp)
    cp.start()

    op = Process(target=output_calc, args=(data_dict,))
    processes.append(op)
    op.start()

    listen(p, output_stream, files, words, '../../../../../../../../../Volumes/External Storage/Thesis/Corpus/eng-balm-emmanuel/flac/')

    for proc in processes:
        proc.join()

