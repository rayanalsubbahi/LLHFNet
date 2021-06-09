import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
from utils import refine_image

device = torch.device(f'cuda:{0}')

class EnhCls(nn.Module):
  def __init__(self):
    super(EnhCls, self).__init__()
    self.res = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    self.res.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    
    self.res2 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    self.res2.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    #enhancer#
    features = []
    features.extend([nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) #131072
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) #131072
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) #131072
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) #131072
    features.extend([nn.AdaptiveAvgPool2d((1, 2))])
    features.extend([nn.Sigmoid()]) 
    self.enhance = nn.Sequential (*features)

    #classifiers#
    self.res.fc = nn.Linear(in_features=2048, out_features=20, bias=True)
    features = []
    features.extend([nn.Dropout(0.3)]) 
    features.extend([nn.Linear(in_features=2048, out_features=20, bias=True)])
    self.res2.fc = nn.Sequential (*features)

  def ftExtract (self, image):
    x = self.res.conv1(image)
    x = self.res.bn1(x)
    x = self.res.relu(x)
    x = self.res.maxpool(x)
    x = self.res.layer1(x)
    x = self.res.layer2(x)
    x = self.res.layer3(x)
    x = self.res.layer4(x) 
    return x
  
  def ftExtract2 (self, image):
    x = self.res2.conv1(image)
    x = self.res2.bn1(x)
    x = self.res2.relu(x)
    x = self.res2.maxpool(x)
    x = self.res2.layer1(x)
    x = self.res2.layer2(x)
    x = self.res2.layer3(x)
    x = self.res2.layer4(x) 
    return x

  def forward(self, image):
    #features
    ft = self.ftExtract (image)

    ##enhancer
    filterHF = self.enhance (ft)
    filterHF = filterHF.reshape (image.size(0),2)
    enhanced_image = refine_image(image,filterHF,device)

    en = self.ftExtract2 (enhanced_image)
    ens = ft + en
    #classifier 2
    cls2 = self.res2.avgpool (ens)
    cls2 = cls2.view(cls2.size(0), -1)
    cls2 = self.res2.fc(cls2)
    #classifier 1
    cls1 = self.res.avgpool (ft)
    cls1 = cls1.view(cls1.size(0), -1)
    cls1 = self.res.fc(cls1)
    return cls2,cls1,enhanced_image,filterHF
