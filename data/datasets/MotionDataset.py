# encoding: utf-8
"""
@author:  Huy Q Can
@contact: huysk82000@gmail.com
"""

import torch
import torch.utils.data as data
import numpy as np
import os
import json

from PIL import Image

class MotionDataset(data.Dataset):
    def __init__(self, 
                 root, 
                 ann_file, 
                 transforms=None) -> None:
        super().__init__()
        self.root = root
        self.img_dir = os.path.join(self.root, 'motion_gradient_map')
        self.transforms = transforms
        
        self.dataset = json.load(open(ann_file, 'r'))
        print('Create dataset for Motion VAE')
    
    def __getitem__(self, index):
        img_path = self.dataset['motion']
        img = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, index
    
    def __len__(self):
        return len(self.dataset['annotaton'])
    
    def get_id_info(self, index):
        image_id = self.dataset['annotations'][index]['image_id']
        track_id = self.dataset['annotations'][index]['track_id']
        
        return image_id, track_id
        