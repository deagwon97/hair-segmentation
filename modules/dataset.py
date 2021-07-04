# -*- coding: utf-8 -*-
from PIL import Image

# for train first model
import torch
import numpy as np
import albumentations as albu
import numpy as np
import torch
import warnings
warnings.filterwarnings(action='ignore')

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class HairDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 name_list = None,
                 preprocessing = None,
                 dir_path = None,
                 augmentation = None, 
                 imgsize = 512,
                 train = True
                ):
        self.name_list = name_list
        self.train = train
        self.dir_path = dir_path
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.imgsize = imgsize
        self.resize =  albu.Compose([
                                  albu.Resize(height = self.imgsize, width = self.imgsize),
                               ])
        self.to_tensor = albu.Compose([
                                  albu.Lambda(image=  to_tensor, mask=to_tensor),
                               ])
    def __len__(self):
        return len(self.name_list)
    
    
    def __getitem__(self,index):
        # 읽어오기 -> numpy변환 -> 가운데만 선택
        image_name = self.name_list[index]
        if self.train == True:
            image_path = self.dir_path + "images/" + image_name + ".jpg"
            mask_path = "./data/mask/" + image_name + ".jpg"

        img = Image.open(image_path)
        img = np.array(img)
        mask = Image.open(mask_path)
        mask = np.array(mask)

        img = (img / 255)
        mask = mask[..., np.newaxis] / 255    

        #apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        # reshape for converting to tensor (모두 적용)
        sample = self.to_tensor(image=img, mask=mask)
        img, mask = sample['image'], sample['mask']
        return img, mask

    
class HairDatasetName(torch.utils.data.Dataset):
    def __init__(self, 
                 name_list = None,
                 preprocessing = None,
                 dir_path = None,
                 augmentation = None, 
                 imgsize = 512,
                 train = True
                ):
        self.name_list = name_list
        self.train = train
        self.dir_path = dir_path
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.imgsize = imgsize
        self.resize =  albu.Compose([
                                  albu.Resize(height = self.imgsize, width = self.imgsize),
                               ])
        self.to_tensor = albu.Compose([
                                  albu.Lambda(image=  to_tensor, mask=to_tensor),
                               ])
    def __len__(self):
        return len(self.name_list)
    
    
    def __getitem__(self,index):
        # 읽어오기 -> numpy변환 -> 가운데만 선택
        image_name = self.name_list[index]
        if self.train == True:
            image_path = self.dir_path+"images/" + image_name + ".jpg"
            mask_path = "./data/mask/" + image_name + ".jpg"

        img = Image.open(image_path)
        img = np.array(img)
        mask = Image.open(mask_path)
        mask = np.array(mask)

        img = (img / 255)
        mask = mask[..., np.newaxis] / 255    

        #apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        # reshape for converting to tensor (모두 적용)
        sample = self.to_tensor(image=img, mask=mask)
        img, mask = sample['image'], sample['mask']
        return img, mask, image_name
