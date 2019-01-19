from helpers import *

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

def generate_data_file(name, filedir, used_classes, unused_classes, order_to_use):
    ## Load all data from given classes in "order_to_use" and write to file, 
    ## dicatated by "name"
    
    used_f = open(filedir + '/sampledFiles.txt', "r")
    used_files = used_f.read().split('\n')
    unused_f = open(filedir + '/unsampledFiles.txt', "r")
    unused_files = unused_f.read().split('\n')
    
    used_length = len(used_classes)
    unused_length = len(unused_classes)
    
    match_matrix = load_matches()
    
    open(filedir + name, 'w').close()  # Delete previous info

    with open(filedir + name, "a") as training_file:
        for i in range(len(order_to_use)):
            to_use_index = order_to_use[i]
            if to_use_index >= used_length:
                info = generate_matched_stream(unused_files, match_matrix, unused_classes[to_use_index - used_length], 
                                               to_use_index - used_length)  # Reset counter for second array
            else:
                info = generate_used_stream(used_files, used_classes[to_use_index], to_use_index)
            to_write = info[0] + " -_- " + info[1] + " -_- " + info[2] + " \n"
            training_file.write(to_write)
            
def generate_used_stream(used_files, class_to_use, to_use):
    ## Generate list structure for exact matches
    
    to_check = [used_files[class_to_use], str(to_use), used_files[class_to_use]]
    return to_check

def generate_matched_stream(unused_files, match_matrix, class_to_use, to_use):
    ## Generate list structure for approximate matches
    
    to_check = [unused_files[class_to_use], str(to_use), match_matrix[class_to_use][2]]
    return to_check
