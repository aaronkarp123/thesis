import os

cpt = 0
for r,d,files in os.walk(os.getcwd()):
	cpt += sum(f.endswith("flac") for f in files)

print(cpt)