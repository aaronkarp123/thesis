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


CHUNK = 1024  # CHUNKS of bytes to read each time from mic
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050

RUNNING = Queue()
RUNNING.put(True)

audioq = Queue()
prespecQ = Queue()
output = Queue()
to_play = Queue()

processes = []

def extract_class_info(matrix):
    data_dict = {}
    os.chdir('../../../../../../../../../Volumes/External Storage/Thesis/Corpus')
    print("Loading audio files...")
    for i in range(len(audio_matrix)):
        filename = audio_matrix[i][0]
        file_class = int(audio_matrix[i][1])
        if file_class not in data_dict:
            yt,sr = librosa.load(filename)
            y, idx = librosa.effects.trim(yt, top_db=50)
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
    while to_run:
        while not output.empty():
            new_temp_out = output.get()
            if new_temp_out == -2:
                to_run = False
            if new_temp_out != cur_out and new_temp_out >= 0:
                print(new_temp_out)
                sys.stdout.flush()
                start_pos = 0
                audio_file = data_dict[new_temp_out]
                while not to_play.empty():
                    to_play.get()
        cur_out = new_temp_out
        if cur_out == -1:
            continue
        if to_play.empty():
            if start_pos+CHUNK > len(audio_file):
                start_pos = 0
            data = audio_file[start_pos:start_pos+CHUNK].tobytes()
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
            audio_data.append(np.fromstring(audioq.get(), dtype=np.float32))
        if len(audio_data) >= time_window and prespecQ.empty():  # If you have a full time-window's worth of data
            audio_data = audio_data[-time_window:]  # Take the most recent time_window chunks
            audio_data_np = np.concatenate(np.array(audio_data)).ravel()
            rms = np.sqrt(np.mean(audio_data_np**2))
            if (rms > 0.02 and currently_above_threshold) or (rms > 0.05 and not currently_above_threshold):
                window = np.pad(audio_data_np, (0, 2048), 'constant', constant_values=(0.0,0.0))
                prespecQ.put(window)
                currently_above_threshold = True
            else:
                currently_above_threshold = False
                output.put(-1)
        if not RUNNING.empty():
            to_run = RUNNING.get()

    print("Process Terminated")

def listen(p, input_stream, output_stream):
    global RUNNING, audioq, output, to_play, CHUNK

    print("** Stream Opened")

    audio_string_chunk = ''

    num_loops = 0
    to_run = True
            
    fe
    
    while to_run:
        try:
            audio_string_chunk = input_stream.read(CHUNK)
        except Exception as e:
            print("Dropped Chunk")
            continue
        
        audioq.put(audio_string_chunk)

        if not prespecQ.empty():
            ad = prespecQ.get()
            spec = get_spectrogram(ad, 22050, n_mels=128, display=False)
            transposed = spec.T
            transposed = scaler.fit_transform(transposed)
            output.put(model.predict_classes(np.array([transposed[0:10]]), batch_size=1)[0])
        
        if not to_play.empty():
            output_stream.write(to_play.get(), CHUNK)

        if (num_loops == 200):
            RUNNING.put(False)
            to_run = False
            output.put(-2)

        num_loops += 1

    print("** Stream Terminated")

    input_stream.close()
    output_stream.close()
    p.terminate()


if __name__ == '__main__':

    audio_matrix = load_stream(name="/trainingInfo_100.txt", filedir='lstmData')
    old_dir = os.getcwd()
    os.chdir('../../../../../../../../../Volumes/External Storage/Thesis/Corpus')
    data_dict = extract_class_info(audio_matrix)
    os.chdir(old_dir)

    model = load_model('lstmData/LSTMModel_100.h5')

    p = pyaudio.PyAudio()
    input_stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)
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

    listen(p, input_stream, output_stream)

    for proc in processes:
        proc.join()

