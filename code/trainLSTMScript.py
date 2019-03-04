from helpers import *
from lstmHelpers import *

import random
from random import randrange, shuffle
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import time
from collections import deque
import pickle
import sys, getopt

filedir = "lstmData"

def generate_data_to_store(NUM_QUERY_FILES):
    num_files_in_stream = 20 * NUM_QUERY_FILES
    order_to_use = []

    used_classes, unused_classes = load_classes("../", NUM_QUERY_FILES, display=False)

    for i in range(NUM_QUERY_FILES):
        order_to_use.extend([i] * int(num_files_in_stream / NUM_QUERY_FILES))
        
    shuffle(order_to_use)
    generate_data_file("/trainingInfo_" + str(NUM_QUERY_FILES) + ".txt", filedir, used_classes, unused_classes, order_to_use, max_ramp_length=0.25)

def build_data_from_file(NUM_QUERY_FILES):
    audio_matrix = load_stream(name="/trainingInfo_" + str(NUM_QUERY_FILES) + ".txt", filedir=filedir)
    composite_signal, composite_matches, preloaded_data = generate_composite_stream(audio_matrix)

    audio_matrix = []
    audio_matrix_test = []
    preloaded_data = {}

    NUM_CLASSES_USED = len(set(composite_matches))
    return composite_signal, composite_matches, NUM_CLASSES_USED

def train_lstm(NUM_QUERY_FILES, NUM_CLASSES_USED, composite_signal, composite_matches):
    n_mels = 128
    seq_length = 10

    start_point = 0
    batch_length = 5000  # In # of sequences
    audio_window= 64
    batch_window = batch_length * audio_window

    lstm_out = 350
    batch_size = 32
    dropout = 0.1
    dropout_r = 0.1
    number_outputs = NUM_CLASSES_USED

    model = Sequential()
    model.add(LSTM(lstm_out, input_shape=(seq_length, n_mels), dropout = dropout, recurrent_dropout = dropout_r))
    model.add(Dense(number_outputs,activation='softmax'))  #softmax
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])  # sparse_categorical_crossentropy
    print(model.summary())

    composite_signal_length = len(composite_signal)

    accs = []
    acc_0 = []
    acc_1 = []
    acc_2 = []
    acc_3 = []

    while start_point + batch_window < composite_signal_length:
        print("Training Data: " + str(start_point) + " / " + str(composite_signal_length) + " ~= " + str(round(start_point / composite_signal_length * 100)) + "%")
        training_data, training_classes = batch(composite_signal[start_point:start_point+batch_window], composite_matches[start_point:start_point+batch_window])
        training_data = np.array(training_data)
        history = model.fit(training_data, training_classes, epochs=4, batch_size=batch_size, verbose=0, shuffle=True)
        accuracies = history.history['acc']
        acc_0.append(accuracies[0])
        acc_1.append(accuracies[1])
        acc_2.append(accuracies[2])
        acc_3.append(accuracies[3])
        os.write(1, str(str(round(start_point / composite_signal_length * 100)) + "\n").encode())
        start_point += batch_window

    model.save(filedir + "/LSTMModel_" + str(NUM_QUERY_FILES) + ".h5")

    accs.append(acc_0)
    accs.append(acc_1)
    accs.append(acc_2)
    accs.append(acc_3)
    return accs

def plot_lstm(accs):
    from matplotlib import pyplot as mp
    acc_0 = accs[0]
    acc_1 = accs[1]
    acc_2 = accs[2]
    acc_3 = accs[3]
    plt.figure(figsize = (15,11))
    plt.plot( np.arange(len(acc_0)), acc_0, label='Epoch 1')
    plt.plot( np.arange(len(acc_1)), acc_1, label='Epoch 2')
    plt.plot( np.arange(len(acc_2)), acc_2, label='Epoch 3')
    plt.plot( np.arange(len(acc_3)), acc_3, label='Epoch 4')
    plt.legend()

    mp.savefig(filedir + '/accuracies_' + str(NUM_QUERY_FILES) + '.png', bbox_inches='tight')

if __name__ == "__main__":
    args = sys.argv
    print("Num Classes: " + args[1])
    NUM_QUERY_FILES = int(args[1])
    generate_data_to_store(NUM_QUERY_FILES)
    composite_signal, composite_matches, NUM_CLASSES_USED = build_data_from_file(NUM_QUERY_FILES)
    accs = train_lstm(NUM_QUERY_FILES, NUM_CLASSES_USED, composite_signal, composite_matches)
    plot_lstm(accs)

