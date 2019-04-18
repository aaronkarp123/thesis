
def split_files(x):
	t = []
	w = []
	for line in x:
		pieces = line.split(":")
		t.append(pieces[0].strip())
		w.append(pieces[1].strip())
	return t, w

def compare_two_files(file1, file2, dire):
	x1 = open(dire+file1, 'r')
	x2 = open(dire+file2, 'r')
	t1, w1 = split_files(x1)
	t2, w2 = split_files(x2)
	print(len(t1))
	print(len(t2))
	correct = 0
	incorrect = 0
	for i in range(len(w1)):
		if w1[i].strip() == w2[i].strip():
			correct += 1
		else:
			incorrect += 1
	print(correct)
	print(incorrect)
	print(correct / (correct + incorrect))

f1 = "timestamp_test_True2.txt"
f2 = "timestamp_test_ASR_True2.txt"
compare_two_files(f1, f2, "Testing/")