# -*- coding: utf-8 -*-
import json
import tqdm
import os
import numpy as np
from PIL import Image, ImageDraw

def generate_mask():
    os.makedirs("./data/mask", exist_ok=True)
    with open("/DATA/Final_DATA/task02_train/labels.json", "r") as json_file:
        labels = json.load(json_file)
    data_num = len(labels['annotations'])

    for idx in tqdm.tqdm(range(data_num), total = data_num):
        polygon_dict = labels['annotations'][idx]['polygon1']
        file_path = labels['annotations'][idx]['file_name']
        polygon = []
        for point in polygon_dict:
            polygon.append(tuple(point.values()))
        img = Image.new('L', (512, 512), 'black')
        ImageDraw.Draw(img).polygon(polygon, outline='white', fill='white')
        np.array(img)
        img.save(os.path.join("./data/mask", file_path.split('.')[0]+'.jpg'))
