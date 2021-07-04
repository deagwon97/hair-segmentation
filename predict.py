# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import os
import random
import json
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings(action='ignore')
from modules.utils import seed_everything


def select_contour(contours):
    # contour가 2개 이상일 경우
    # 가장 point 수가 많은 contour를 mask로 사용

    largest_contour_id = 0
    max_point_counts = 0

    for contour_id, contour in enumerate(contours):
        if len(contour) > max_point_counts:
            largest_contour_id = contour_id
            max_point_counts = len(contour)
        else:
            pass
        
    return largest_contour_id

def contour_to_mask(polygon):
    temp = []

    for point in polygon:
        temp.append(tuple(point.values()))

    mask_img = Image.new('L', (512, 512), 'black')
    ImageDraw.Draw(mask_img).polygon(temp, outline='white', fill='white')
    return mask_img

def set_form(contour):
    polygon1 = []
    
    for _, coord in enumerate(contour):
        x = coord[0,0]
        y = coord[0,1]
        polygon1.append({"x":int(x), "y":int(y)})
    return polygon1

def get_iou(mask1, mask2):
    union = np.logical_or(mask1, mask2)
    intersection = np.logical_and(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def compare_contour(sample_pred, threshold = 0.994026):
    contours1, _  = cv2.findContours(sample_pred, 
                                cv2.RETR_EXTERNAL, 
                                #cv2.CHAIN_APPROX_SIMPLE)
                                 cv2.CHAIN_APPROX_NONE) # 516개 point 생성 -> 총 json 파일 용량이 200MB를 초과
                                # cv2.CHAIN_APPROX_TC89_L1) # 171개 point 생성 -> 52MB 초과
                                # cv2.CHAIN_APPROX_TC89_KCOS) #130개 point 생성

    contours2, _  = cv2.findContours(sample_pred, 
                    cv2.RETR_EXTERNAL, 
                    # cv2.CHAIN_APPROX_SIMPLE)
                    # cv2.CHAIN_APPROX_NONE) # 516개 point 생성 -> 총 json 파일 용량이 200MB를 초과
                    # cv2.CHAIN_APPROX_TC89_L1) # 171개 point 생성 -> 52MB 초과
                    cv2.CHAIN_APPROX_TC89_KCOS) #130개 point 생성

    if len(contours1) >= 2:
        largest_contour_id1 = select_contour(contours1)
    else:
        largest_contour_id1 = 0

    if len(contours2) >= 2:
        largest_contour_id2 = select_contour(contours2)
    else:
        largest_contour_id2 = 0

    contour1 = contours1[largest_contour_id1]
    contour2 = contours2[largest_contour_id2]

    polygon1 = set_form(contour1)
    polygon2 = set_form(contour2)

    mask1 = contour_to_mask(polygon1)
    mask2 = contour_to_mask(polygon2)


    # 두 알고리즘의 iou 차이가 threshold 이상인 이미지는 1 알고리즘(손실 적음), 아닌 이미지는 2 알고리즘
    diff = get_iou(mask1, mask2)

    if diff < threshold:
        polygon = polygon1
        #print("CHAIN_APPROX_NONE")
    else:
        polygon = polygon2 

    return polygon, diff , polygon1, polygon2


if __name__ == "__main__":

    DATA_PATH = "/DATA/FINAL_DATA"
    
    with open(DATA_PATH + "/task02_test/sample_submission.json", "r") as json_file:
        labels = json.load(json_file)

    seed_everything(1015)

    with torch.no_grad():
        PATH_1 = "./model/first_model.pth"
        model_1 = torch.load(PATH_1, map_location = "cuda")
        PATH_2 = "./model/second_model.pth"
        model_2 = torch.load(PATH_2, map_location = "cuda")

    diff_array = []
    polygon1s = []
    polygon2s = []
    for idx, item_name in tqdm.tqdm(enumerate(labels['annotations']),
                        total = len(labels['annotations'])):

        sample_img = Image.open(DATA_PATH + "/task02_test/images/" + item_name['file_name'])
        sample_img = np.array(sample_img) / 255
        sample_img = sample_img.transpose(2, 0, 1).astype('float32')

        sample_tensor_1 = torch.cuda.FloatTensor(sample_img[np.newaxis,...], 
                        device = "cuda")
        sample_tensor_2 = torch.cuda.FloatTensor(sample_img[np.newaxis,...], 
                        device = "cuda")
        
        sample_pred_1   = model_1(sample_tensor_1).cpu().detach().numpy()[0,0,...]
        sample_pred_2   = model_2(sample_tensor_2).cpu().detach().numpy()[0,0,...]
        
        sample_pred = (sample_pred_1 + sample_pred_2) / 2
        
        sample_pred   = (sample_pred > 0.5) * 1

        sample_pred = sample_pred.astype(np.uint8)
        
        polygon, diff, polygon1, polygon2 = compare_contour(sample_pred,  
                                                            threshold = 0.98999)
                                        
        diff_array.append(diff)
        polygon1s.append(polygon1)
        polygon2s.append(polygon2)

        labels['annotations'][idx]['polygon1'] = polygon
        
    threshold = np.quantile(diff_array, 0.04).round(5)
    print(threshold)
    
    for idx, item_name in tqdm.tqdm(enumerate(labels['annotations']),
                        total = len(labels['annotations'])):
        if diff_array[idx] < threshold:
            polygon = polygon1s[idx]
        else:
            polygon = polygon2s[idx]
        labels['annotations'][idx]['polygon1'] = polygon

    with open("./output/submission.json", "w") as json_file:
        json.dump(labels, json_file)