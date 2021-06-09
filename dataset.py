import glob
import os
import random

import numpy as np
import cv2
from torch.utils.data import Dataset

class SICEPart1(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.low_dir = img_dir
        self.high_dir = gt_dir
        self.images = [im_name for im_name in os.listdir(img_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        ###LLIs###
        low = cv2.imread(os.path.join(self.low_dir, name))
        lowImg = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)
        lowImg = lowImg [:,:,2]

        ###NLIs###
        imHighName = name[:name.index('_')]+'.JPG'
        high = cv2.imread(os.path.join(self.high_dir, imHighName)) 
        highImg = cv2.cvtColor(high, cv2.COLOR_BGR2HSV)
        highImg = highImg [:,:,2]

        if self.transform:
            lowImg = self.transform(lowImg)
            highImg = self.transform (highImg)

        sample = {'name': name, 'lowImg': lowImg, 'highImg': highImg, 'low': low, 'high': high}
        return sample
