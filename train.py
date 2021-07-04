# -*- coding: utf-8 -*-
# for gen data
import os
# for train first model
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
import torch
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings(action='ignore')
import os

# import custom utils
from modules.utils import gen_cleandf
from modules.utils import seed_everything
from modules.train_fold import run
from modules.gen_mask import generate_mask

if __name__ == "__main__":

    # make mask-----------------------------------------------------
    print("convert json to jpg...")
    generate_mask()
    DATA_PATH = "/DATA/Final_DATA"
    # set image list-------------------------------------------------
    seed_everything(1015)
    trainlist = list(map(lambda x: x[:-4], 
                    os.listdir(DATA_PATH + "/task02_train/images")))
    drop_ipynb = []
    for item in trainlist:
        if 'ipynb' not in item:
            drop_ipynb.append(item)
    trainlist = drop_ipynb
    trainlist.sort()
    
    # set config-----------------------------------------------------
    CONFIG = {}
    CONFIG['augmentation']= [
                                albu.Transpose(p=0.5),
                                albu.RandomRotate90(3),
                                albu.Rotate(p=1, border_mode = 1),
                                albu.GridDistortion(p = 0.1), # 2배
                                albu.Cutout(p=0.1, 
                                            num_holes=30, 
                                            max_h_size=20, 
                                            max_w_size=20),
                                ]
    CONFIG['fold']= 1
    CONFIG['num_epochs']= 6
    CONFIG['batch_size']= 20
    CONFIG['model'] = "efficientnet-b4"
    CONFIG['pretrain'] = "imagenet"
    CONFIG['optimizer']= torch.optim.Adam
    CONFIG['scheduler']= lr_scheduler.CosineAnnealingWarmRestarts
    CONFIG['loss']= smp.utils.losses.DiceLoss
    CONFIG['device']= "cuda"
    CONFIG['model_path'] = "./model"

    seed_everything(1015)
    print("train first model...")
    run(trainlist, CONFIG, "first_model.pth")
    
    # predict frist model--------------------------------------------
    print("get iou score ...")
    model_path = './model/first_model.pth'
    clean_df = gen_cleandf(model_path, 
                              DATA_PATH+"/task02_train/", 
                              CONFIG['device'], 
                              trainlist)
    print(clean_df.shape)
    seed_everything(1015) 
    # run second_model
    trainlist = clean_df.image_name.values
    CONFIG['fold']= 0
    print("train second model...")
    run(trainlist, CONFIG, "second_model.pth")
    print("complete.")
