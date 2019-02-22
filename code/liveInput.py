import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue
import time
import sys
import atexit

CHUNK = 1024  # CHUNKS of bytes to read each time from mic
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050

RUNNING = Queue()
RUNNING.put(True)

audioq = Queue()

processes = []

def concur_process():
    global RUNNING, audioq

    audio_data = []
    time_window = (2048*3) / CHUNK

    while RUNNING.empty() or RUNNING.get():
        while not audioq.empty():
            audio_data.append(np.fromstring(audioq.get(), dtype=np.float32))
        if len(audio_data) >= time_window:  # If you have a full time-window's worth of data
            audio_data = audio_data[-time_window:]  # Take the most recent time_window chunks 
            window = np.pad(np.concatenate(np.array(audio_data)).ravel(), (0, 2048), 'constant', constant_values=(0.0,0.0))

    print("Process Terminated")

def listen(p, stream):
    global RUNNING, qudioq, CHUNK

    print("** Stream Opened")

    audio_string_chunk = ''

    num_loops = 0
    to_run = True
    while to_run:
        audio_string_chunk = stream.read(CHUNK)

        audioq.put(audio_string_chunk)

        if (num_loops == 100):
            RUNNING.put(False)
            to_run = False

        num_loops += 1

    print("** Stream Terminated")

    stream.close()
    p.terminate()


p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)

cp = Process(target=concur_process, args=())
processes.append(cp)
cp.start()

listen(p, stream)

for proc in processes:
    proc.join()

