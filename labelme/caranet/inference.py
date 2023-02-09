import numpy as np
import pandas as pd
import torch
import os
import cv2
import argparse
import sys
from caranet.utils.dataloader import get_loader, test_dataset, inference_dataset
import torch.nn.functional as F
import numpy as np
from caranet.CaraNet import caranet
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import labelme
import base64

save_imgs = True # if False, then just predict segmentation masks, otherwise predict masked image

'''
Input: Path to folder containing images
Output: Predicted Segmentation mask

Check:
Should work for both CPU and GPU
Post process the small image blobs
'''

class Inference_Villi:
    def __init__(self, model_path):
        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model = self.load_model()

    def load_model(self):
        model = caranet()
        weights = torch.load(self.model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            if 'total_ops' not in k and 'total_params' not in k:
                name = k
                new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(device=self.device)
        return model

    def remove_small_blob(self, res):
        im_gauss = cv2.GaussianBlur(res, (5, 5), 0)
        ret, thresh = cv2.threshold(im_gauss, 127, 255, 0)
        thresh = thresh.astype(np.uint8)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_area = []
        #create a mask
        mask = np.ones(res.shape[:2], dtype="uint8") * 255
        # calculate area and filter into new array
        for con in contours:
            area = cv2.contourArea(con)
            if 500 < area < 10000:
                cv2.drawContours(mask, [con], -1, 0, -1)
                contours_area.append(con)
        process_image = cv2.bitwise_and(res, res, mask=mask)
        return process_image


    def get_predictions(self, image_path):
        #load single image
        inference_loader = inference_dataset(image_path)
        image = inference_loader.load_data()
        if self.device == torch.device('cuda:0'):
            image = image.cuda()
        res5,res4,res2,res1 = self.model(image)
        res = res5
        res = F.upsample(res, size= (640, 640), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res>0.3).astype('float')
        res = res * 255
        image = image.squeeze()
        image = image.permute(1,2,0)
        image = image.cpu().detach().numpy()
        image = (image - image.min())/(image.max() - image.min())
        process_image = self.remove_small_blob(res)
        return process_image



def infer(img_location, return_type = 'json'):
    #checking the inference
    model_save_path = '../../weights/CaraNet-best_84.pth'
    temp_save_dir = './caranet/results/interpretable_images/'
    
    # Jsonify Output File
    

    image_path = img_location
    Inference_villi = Inference_Villi(model_save_path)
    predicted_mask = Inference_villi.get_predictions(image_path)
    
    ## -------- ##
    json_saver = {}
    json_saver["imageHeight"] = 640
    json_saver["imageWidth"] = 640
    json_saver["version"] =  "5.1.1"
    json_saver["flags"] = {}
    json_saver["shapes"] = []
    data = labelme.LabelFile.load_image_file(image_path)
    image_data = base64.b64encode(data).decode('utf-8')
    json_saver["imageData"] = image_data
    ## -------- ##

    if save_imgs:
        
        

        img = cv2.imread(image_path)
        predicted_mask = (predicted_mask[:, :] > 0.1) * 255
        #####
        contour_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
        for x in range(predicted_mask.shape[0]):
            for y in range(predicted_mask.shape[1]):
                contour_mask[x][y] = predicted_mask[x][y]
        
        ret, thresh = cv2.threshold(contour_mask, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #######
        
        predicted_mask = np.expand_dims(predicted_mask // 255, axis=-1)
        
        shapes = []

        for hull in contours:
            shape = {}
            shape['label'] = 'Inference'
            shape["line_color"] = [
                0,
                10,
                0,
                20
            ],
            shape["fill_color"] =  [
                0,
                20,
                0,
                40
            ]
            shape['points'] = []

            hull = np.reshape(hull,(hull.shape[0],2))
            hull = hull[::6,:] # take every 6th point only
            for point in hull:
                add = [int(point[0]),int(point[1])]
                shape["points"].append(add)

            shape["group_id"] = None,
            shape["shape_type"] =  "polygon"
            shape["flags"] =  {}
            shapes.append(shape)

        img = img*predicted_mask
        img_name = img_location.split('/')[-1]
        cv2.imwrite(os.path.join(temp_save_dir,img_name),img)    
        image_path =  os.path.join(temp_save_dir,img_location)
        json_saver["imagePath"] = image_path
        
    else:
        cv2.imwrite(os.path.join(temp_save_dir,img_location),predicted_mask)

    json_saver["shapes"] = shapes
    json_object = json.dumps(json_saver)     

    with open('./caranet/results/annotations/'+img_location.split('/')[-1].split('.')[0]+'.json','w') as f:
        f.write(json_object)
    
    f.close()

    if return_type == 'img':
        return os.path.join(temp_save_dir,img_name)

    return './caranet/results/annotations/'+img_location.split('/')[-1].split('.')[0]+'.json'


