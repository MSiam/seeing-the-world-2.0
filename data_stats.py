import os
import argparse
import matplotlib.pyplot as plt
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--main_dir', type=str, default='')
args = parser.parse_args()

main_dir = args.main_dir

nclasses = {}

for cls in os.listdir(main_dir):
    if cls[0] == '.':
        continue
    files = glob.glob(main_dir+cls+'/*.JPG')
    files.extend(glob.glob(main_dir+cls+'/*.jpg'))
    files.extend(glob.glob(main_dir+cls+'/*.png'))
    nclasses[cls] = len(files)

#plt.hist(nclasses.values(), bins=range(len(nclasses.keys())))
plt.plot(range(len(nclasses.keys())), [*nclasses.values()])
plt.xticks(range(len(nclasses.keys())), nclasses.keys(), rotation='vertical')
#plt.show()

# Fewshot classes
import pdb; pdb.set_trace()
clsses = np.array([*nclasses.keys()])
ns = np.array([*nclasses.values()])
print('Few-shot Classes with less than 20 samples : ', clsses[ns<20])
# 64 Total Classes
# 37 Classes with Fewshot data <20


