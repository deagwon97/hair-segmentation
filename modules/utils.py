# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image, ImageDraw
import tqdm
import cv2
import pandas as pd
import os
import random
from sklearn.model_selection import KFold
from tqdm import tqdm

import os
import sys
modulde_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(modulde_path)

from dataset import HairDatasetName




def get_iou(grd, pred):
    grd = grd.reshape(-1)
    pred = pred.reshape(-1)
    union = (grd + pred) > 0
    intersect = (grd + pred) >= 2
    iou = intersect.sum() / union.sum()
    return iou

def gen_cleandf(model_path, 
              imag_dir_path, 
              device, 
              trainlist):
    
    model = torch.load(model_path, map_location = device)
    model.eval()
    
    target_fold_index = 0
    kfold = KFold(n_splits=5, 
                  random_state = 1015, 
                  shuffle = True)
    trainlist = np.array(trainlist)
    for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(trainlist)):
        if fold_index == target_fold_index:
            train_fold = trainlist[trn_idx]
            valid_fold = trainlist[val_idx]
            
            train_dataset = HairDatasetName(name_list = train_fold,
                train = True,
                dir_path = "/DATA/FINAL_DATA/task02_train/",
                preprocessing=None,
                augmentation=None)

            valid_dataset = HairDatasetName(name_list = valid_fold,
                train = True,
                dir_path = "/DATA/FINAL_DATA/task02_train/",
                preprocessing=None,
                augmentation=None)
           
            valid_scores = []
            for (sample_img, mask, image_name) in tqdm(valid_dataset,
                                        total = valid_dataset.__len__()):
                sample_tensor = torch.cuda.FloatTensor(sample_img[np.newaxis,...], 
                                device = device)
                sample_pred   = model(sample_tensor).cpu().detach().numpy()[0,0,...]
                sample_pred   = (sample_pred > 0.5) * 1
                iou = get_iou(sample_pred, mask)
                valid_scores.append([image_name, iou])
                
            
            train_scores = []
            for (sample_img, mask, image_name) in tqdm(train_dataset,
                                        total = train_dataset.__len__()):
                sample_tensor = torch.cuda.FloatTensor(sample_img[np.newaxis,...], 
                                device = device)
                sample_pred   = model(sample_tensor).cpu().detach().numpy()[0, 0, ...]
                sample_pred   = (sample_pred > 0.5) * 1
                iou = get_iou(sample_pred, mask)
                train_scores.append([image_name, iou])
            
                
    train_score = pd.DataFrame(train_scores, 
                                columns = ['image_name', 'iou'])
    valid_score = pd.DataFrame(valid_scores, 
                                columns = ['image_name', 'iou'])
    
    total_scores = pd.concat([train_score, valid_score], axis = 0)
    clean_df = total_scores[total_scores.iou > 0.7]
    #clean_df = total_scores#[total_scores.iou > 0.001]
    return clean_df


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.
    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  