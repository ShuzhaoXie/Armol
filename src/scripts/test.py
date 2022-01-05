import numpy as np
import json

with open('train_rank_2.json') as f:
    lines = f.readlines()
    image_names = []
    for line in lines:
        image_names.append(line.strip())

with open('train_rank.json', 'w') as f:
    json.dump(image_names, f)