import numpy as np
import skimage.io as io
from skimage.transform import resize
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import os
import sys
import json
import pandas as pd
sys.path.append('..')
from mypool import MyPool

df = pd.read_csv('Flickr8k.token.txt', sep='\t')
print(df.head())
annotations = {}


def prc_data(filename):
    img_file = resize(io.imread(os.path.join('img_raw', filename)), (224, 224))

    anns = df[df['id'].str.contains(filename)]['ann'].to_list()
    annotations[filename] = anns
    io.imsave(os.path.join('img_prc', filename), img_as_ubyte(img_file))


def get_cap(filename):
    anns = df[df['id'].str.contains(filename)]['ann'].to_list()
    # print(anns)
    annotations[filename] = anns

_ = [get_cap(x) for x in os.listdir('img_prc') if '.jpg' in x]

# pool = MyPool(15)
# pool.map(prc_data, [x for x in os.listdir('img_raw') if '.jpg' in x])
# pool.close()
# pool.join()

with open('captions.json', 'w') as f:
    json.dump(annotations, f)
