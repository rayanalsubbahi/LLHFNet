import os
import argparse

import numpy as np
import cv2

from model import *
from utils import *

classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='device num, cuda:0')
    parser.add_argument('--test_Dir', type=str, required=True, help='path to test images')
    parser.add_argument('--ckpt_Dir', type=str, required=True, help='path to *model.pth')
    parser.add_argument('--results_Dir', type=str, required=True, help='path to save result images')
    args = parser.parse_args()
    return args

args = parse_args()

if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)

checkpoint = torch.load(args.ckpt_Dir, map_location=device)
hp = checkpoint['HyperParam']
model = EnhCls()
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

imgs = os.listdir(args.test_Dir)

with torch.no_grad():
  for img in imgs:
    img_init = cv2.imread (args.test_Dir + "/" + img)
    h,w,_ = img_init.shape
    img_init = cv2.cvtColor(img_init, cv2.COLOR_BGR2HSV)
    img_LLI = img_init/255.0
    img_batch = torch.from_numpy(img_LLI[:,:,2].reshape(1,1,h,w)).type(torch.FloatTensor).cuda()
    try:
      outputs2,outputs1,enhanced_batch,filterHF = model(img_batch)
    except: 
       continue

    _, predicted = torch.max(outputs2, 1)
    print (img + ": " + classes[predicted.item()])

    enhanced = enhanced_batch.cpu().numpy().reshape(h,w)
    im = img_init 
    im [:,:,2] = enhanced*255
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    cv2.imwrite (args.results_Dir + '/' + img + '.png', im)
