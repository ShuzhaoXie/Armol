import numpy as np
import os
import glob
import json

from numpy.lib.shape_base import split

from common import WORK_DIR


with open(f'{WORK_DIR}/src/scripts/test_rank.json', 'r') as f:
    rank = json.load(f)

with open(f'{WORK_DIR}/src/scripts/word2num.json', 'r') as f:
    syn = json.load(f)
# book-1-493-155-525-162
# dining table-60-1-0-275-640-411

for i, img_name in enumerate(rank):
    file_path = f'{WORK_DIR}/data/test/ground_truth_old/{i}.txt'
    new_file_path = f'{WORK_DIR}/data/test/ground_truth/{i}.txt'
    f = open(new_file_path, 'w')
    with open(file_path, 'r') as gt:
        lines = gt.readlines()
        for line in lines:
            class_name, confidence, left, top, right, bottom = line.strip().split('-')
            class_id = syn[class_name.lower()] 
            line = "-".join([class_name, str(class_id), confidence, left, top, right, bottom]) + "\n"
            print(line)
            f.write(line)
    f.close()
            
