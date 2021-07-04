# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
# import wandb

import os
import sys
modulde_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(modulde_path)

from dataset import HairDataset

def run(trainlist, CONFIG, model_name):
    
    def get_training_augmentation():
        transform = CONFIG['augmentation']
        return albu.Compose(transform)
    
    
    target_fold_index = CONFIG['fold']
    kfold = KFold(n_splits=5, 
                  random_state = 1015, 
                  shuffle = True)
    trainlist = np.array(trainlist)
    for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(trainlist)):
        if fold_index == target_fold_index:
            print(f"start training {model_name} !!")
            train_fold = trainlist[trn_idx]
            valid_fold = trainlist[val_idx]
            ENCODER = CONFIG['model']
            ENCODER_WEIGHTS = CONFIG['pretrain']
            ACTIVATION = 'sigmoid'

            model = smp.Unet(
                encoder_name=ENCODER, 
                encoder_weights=ENCODER_WEIGHTS, 
                in_channels = 3,
                classes=1, 
                activation = ACTIVATION,
            )


            train_dataset = HairDataset(name_list = train_fold,
                train = True,
                dir_path =  "/DATA/FINAL_DATA/task02_train/",
                preprocessing=None,
                augmentation=None)

            valid_dataset = HairDataset(name_list = valid_fold,
                train = True,
                dir_path =  "/DATA/FINAL_DATA/task02_train/",
                preprocessing=None,
                augmentation=None)


            BATCH_SIZE = CONFIG['batch_size']
            train_step_size = train_dataset.__len__() // BATCH_SIZE
            valid_step_size = valid_dataset.__len__() // BATCH_SIZE
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 8)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = 2)

            loss = CONFIG['loss']()

            optimizer = CONFIG['optimizer']([ 
                dict(params=model.parameters(), lr=0.0001),
            ])

            metrics = [
                smp.utils.metrics.IoU(threshold=0.5),
            ]

            scheduler = CONFIG['scheduler'](optimizer, 10, 2, eta_min=1e-6)
            train_epoch = smp.utils.train.TrainEpoch(
                model, 
                loss=loss, 
                metrics=metrics, 
                optimizer=optimizer,
                device=CONFIG['device'],
                verbose=True,
            )

            valid_epoch = smp.utils.train.ValidEpoch(
                model, 
                loss=loss, 
                metrics=metrics, 
                device=CONFIG['device'],
                verbose=True,
            )

            # 한 폴드당 150 Epoch 수행
            NUM_EPOCH = CONFIG['num_epochs']
            min_score = np.Inf
            if not os.path.exists(CONFIG['model_path']):
                subprocess.call(["mkdir",
                                CONFIG['model_path']],
                                shell = False)
            #MODEL = train_name
            patient = 0
            max_valid_iou  = 0
            for i in range(0, NUM_EPOCH):

                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)
                
                #TODO 제거
#                 wandb.log({
#                 "valid" : valid_logs['iou_score'],
#                 "train" : train_logs['iou_score']
#                             })

                scheduler.step()
                if max_valid_iou < valid_logs['iou_score']:
                    max_valid_iou = valid_logs['iou_score']
                    torch.save(model, f"{CONFIG['model_path']}/{model_name}")
                    
    print(f"compelte training {model_name} !!")