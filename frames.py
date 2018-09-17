from os import listdir
from os.path import join

dir_name = './items'
counts = []
for key in listdir(dir_name):
    dir_path = join(dir_name, key)
    print(dir_path)
    
    count = len(list(listdir(dir_path)))
    counts.append(count)

import numpy as np

r = np.mean(counts)
print(r)
