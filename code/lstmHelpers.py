from helpers import *
from pyfftw.interfaces.numpy_fft import rfft, irfft
from sklearn.preprocessing import MinMaxScaler

def load_classes(filedir, num_classes, display=False):
    ## Find random classes proportional to the ratio of used_files to unused_files
    
    f_o = open(filedir + '/sampledFiles.txt', "r")
    used_fs = f_o.read().split('\n')
    uf_o = open(filedir + '/unsampledFiles.txt', "r")
    unused_fs = uf_o.read().split('\n')
    
    used_classes = []
    unused_classes = []
    for i in range(num_classes):
        if random.randint(0,len(used_fs)+len(unused_fs)) > len(used_fs):
            unused_classes.append(random.randint(0, len(unused_fs)))
        else:
            used_classes.append(random.randint(0, len(used_fs)))
    print("Using classes...")
    print("Exact Classes: ", end="")
    print(used_classes)
    print("Matched Classes: ", end="")
    print(unused_classes)
    
    if display:
        for c in used_classes:
            y1,sr1 = librosa.load(used_fs[c])
            print(str(c) + " -- used")
            ipd.display(ipd.Audio(y1, rate = sr1)) # load query file
        for c in unused_classes:
            y1,sr1 = librosa.load(unused_fs[c])
            print(str(c) + " -- matched")
            ipd.display(ipd.Audio(y1, rate = sr1)) # load query file
    return used_classes, unused_classes

def generate_data_file(name, filedir, used_classes, unused_classes, order_to_use, max_ramp_length = 0.25):
    ## Load all data from given classes in "order_to_use" and write to file, 
    ## dicatated by "name"
    
    used_f = open('../sampledFiles.txt', "r")
    used_files = used_f.read().split('\n')
    unused_f = open('../unsampledFiles.txt', "r")
    unused_files = unused_f.read().split('\n')
    
    used_length = len(used_classes)
    unused_length = len(unused_classes)
    
    match_matrix = load_matches()
    
    max_ramp_length = int(22050 * max_ramp_length)
    
    used_dict = {}
    for i in range(len(order_to_use)):
        to_use_index = order_to_use[i]
        if to_use_index < used_length:
            tempname = used_files[used_classes[to_use_index]]
            if tempname not in used_dict:
                used_dict[tempname] = order_to_use[i]
    open(filedir + name, 'w').close()  # Delete previous info
   
    with open(filedir + name, "a") as training_file:
        for i in range(len(order_to_use)):
            to_use_index = order_to_use[i]
            if to_use_index >= used_length:
                if match_matrix[unused_classes[to_use_index - used_length]][2] in used_dict:  # if you match to a used file in the current data set
                    to_use_index = used_dict[match_matrix[unused_classes[to_use_index - used_length]][2]]
                    info = generate_matched_to_used_stream(unused_files, used_files, order_to_use[i] - used_length, to_use_index, match_matrix[unused_classes[order_to_use[i] - used_length]][2])
                else:
                    info = generate_matched_stream(unused_files, match_matrix, unused_classes[to_use_index - used_length],
                                                   to_use_index)  # Reset counter for second array
            else:
                info = generate_used_stream(used_files, used_classes[to_use_index], to_use_index)
            to_write = info[0] + " -_- " + info[1] + " -_- " + info[2] + " -_- " + str(random.randint(0, max_ramp_length)) + " -_- " + str(random.randint(0, max_ramp_length)) + " -_- " + str(random.randint(100, 600) / 10.0) + " -_- " + str(np.random.normal(1, 0.1, 1)[0]) + "\n"
            training_file.write(to_write)

def load_stream(name, filedir='../'):
    training_file = open(filedir + name, "r")
    info = training_file.read().split('\n')
    data_matrix = []
    for inf in info:
        segs = inf.split('-_-')
        if len(segs) > 1:
            try:
                data_matrix.append([segs[0].strip(), segs[1].strip(), segs[2].strip(), segs[3].strip(), 
                                    segs[4].strip(), segs[5].strip(), segs[6].strip()])
            except Exception as e:
                #print(e)
                print(inf)
        else:
            print("\n")
    training_file.close()
    return data_matrix
            
def generate_used_stream(used_files, class_to_use, to_use):
    ## Generate list structure for exact matches
    
    to_check = [used_files[class_to_use], str(to_use), used_files[class_to_use]]
    return to_check

def generate_matched_stream(unused_files, match_matrix, class_to_use, to_use):
    ## Generate list structure for approximate matches
    
    to_check = [unused_files[class_to_use], str(to_use), match_matrix[class_to_use][2]]
    return to_check

def generate_matched_to_used_stream(unused_files, used_files, file_number, used_file_number, matched_class):
    to_check = [unused_files[file_number], str(matched_class), used_files[used_file_number]]
    return to_check

def apply_ramp(y, fadein_len = 0, fadeout_len = 0):
    ## Create a linear ramp for fading in and out
    ## Apply the two fades to "y" and return
    
    fade_in = np.linspace(0.0, 1.0, num=int(fadein_len))
    fade_out = np.linspace(1.0, 0.0, num=int(fadeout_len))
    adjusted_fadein = np.pad(fade_in, (0, y.shape[0] - int(fadein_len)), 'edge')
    adjusted_fadeout = np.pad(fade_out, (y.shape[0] - int(fadeout_len), 0), 'edge')
    if int(fadein_len) == 0:
        fades = adjusted_fadeout
    elif int(fadeout_len) == 0:
        fades = adjusted_fadein
    else:
        fades = np.multiply(adjusted_fadein, adjusted_fadeout)
    return np.multiply(y, fades)

def apply_noise(y, target_snr_db=20):
    ## Adapted from: https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    ## Apply random white noise to signal to reach target_snr_db

    x_watts = y ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - float(target_snr_db)
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, noise_avg_watts, len(x_watts))
    Xb = rfft(noise_volts) / len(noise_volts)
    Sb = np.arange(Xb.size)+1  # Filter
    yb = irfft(Xb/Sb).real[:len(noise_volts)]
    # Noise up the original signal
    noisy_signal = y + yb

    return np.clip(noisy_signal, -1, 1)

def timewarp(sig, factor=1):
    return librosa.effects.time_stretch(sig, float(factor))

def generate_composite_stream(audio_matrix, data_dict = None):
    cur_percentage = 0
    if data_dict == None:
        data_dict = {}
    
    composite_signal_list = []
    composite_matches_list = []
    
    for i in range(len(audio_matrix)):
        if (round(i / len(audio_matrix) * 100) != cur_percentage):
            cur_percentage = round(i / len(audio_matrix) * 100)
            print(str(cur_percentage) + "%     ", end='')
            
        filename = audio_matrix[i][0]
        if filename in data_dict:
            y = data_dict.get(filename)
        else:
            yt,sr = librosa.load(filename)
            y, idx = librosa.effects.trim(yt, top_db=50)
            data_dict[filename] = y
        warped_y = timewarp(y, audio_matrix[i][6])
        noisey_y = apply_noise(warped_y, audio_matrix[i][5])
        faded_y = apply_ramp(noisey_y, audio_matrix[i][3], audio_matrix[i][4])
        composite_signal_list.append(faded_y)
        composite_matches_list.append(np.full(faded_y.shape, audio_matrix[i][1]))
    composite_signal = np.array(composite_signal_list)
    composite_signal = np.concatenate(composite_signal).ravel()
    composite_matches = np.array(composite_matches_list)
    composite_matches = np.concatenate(composite_matches).ravel()
    return composite_signal, composite_matches, data_dict

def batch(signal, matches, hop_length = 512/8, print_output=False):
    signal_batch_length = 2048*3
    data = []
    classes = []
    
    batched_frames = []
    cur_frame_count = 0
    cur_percentage = 0
    num_to_add  = signal_batch_length

    n_mels = 128
    seq_length = 10

    while cur_frame_count < len(signal):
        batched_frames.extend(signal[cur_frame_count : cur_frame_count + num_to_add - 1])
        recent_signal = np.asarray(batched_frames)
        recent_signal = np.pad(recent_signal, (0, 2048), 'constant', constant_values=(0.0,0.0))
        spec = get_spectrogram(recent_signal, 22050, n_mels=n_mels, display=False)
        transposed = spec.T
        scaler = MinMaxScaler(feature_range=(0, 1))
        transposed = scaler.fit_transform(transposed)
        comp_cols = []
        for i in range(0,seq_length):
            comp_cols.append(transposed[i])
        data.append(comp_cols)
        classes.append(int(matches[cur_frame_count-1]))
        batched_frames = batched_frames[int(hop_length)-1:]  # hop length of spectrogram / 8
        
        if print_output and round((cur_frame_count-1) / len(signal) * 100) != cur_percentage:
            cur_percentage = round((cur_frame_count-1) / len(signal) * 100)
            print(str(cur_percentage) + "%     ", end='')
            
        cur_frame_count += num_to_add
        num_to_add = int(hop_length)
        
    return data, classes