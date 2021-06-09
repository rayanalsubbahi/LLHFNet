import os
import argparse

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from model import *
from utils import *
from dataset import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='device num, cuda:0')
    parser.add_argument('--featureExt', type=str, required=True, help='either: resnet50, mobilenetv2, vgg16, densenet, squeezenet')
    parser.add_argument('--test_Dir', type=str, required=True, help='path to test images')
    parser.add_argument('--results_Dir', type=str, required=True, help='path to save result images')
    parser.add_argument('--ckpt_Dir', type=str, required=True, help='path to *model.pth')

    args = parser.parse_args()
    return args


args = parse_args()
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)

print(f'[START EVALUATION JOB] \n')

checkpoint = torch.load(args.ckpt_Dir, map_location=device)
hp = checkpoint['HyperParam']
epoch = checkpoint['epoch']
model = switch_model(args.featureExt)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

imgs = os.listdir (args.test_Dir)

with torch.no_grad():
    for im in imgs:
        img_init = cv2.imread (args.test_Dir + "/" + im)
        h,w,_ = img_init.shape
        img_init = cv2.cvtColor(img_init, cv2.COLOR_BGR2HSV)
        img_LLI = img_init/255.0
        img_batch = torch.from_numpy(img_LLI[:,:,2].reshape(1,1,h,w)).type(torch.FloatTensor).cuda()
        try:
         filterHF = model(img_batch)
        except: 
          continue
        filterHF = filterHF.reshape (img_batch.size(0),2)
        enhanced_batch = refine_image(img_batch, filterHF,device)
        enhanced = enhanced_batch.cpu().numpy().reshape(h,w)
        img = img_init 
        img [:,:,2] = enhanced*255
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        cv2.imwrite (args.results_Dir + '/' + im + '.png', img)
      

