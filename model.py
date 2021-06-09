import torch
import torch.nn as nn
import torch.nn.functional as F

#MobileNetv2
class MobileNetv2(nn.Module):
  def __init__(self):
    super(MobileNetv2, self).__init__()
    self.mbl = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    features = list(self.mbl.features)[1:]
    first_conv_layer = [nn.Conv2d(1, 32, kernel_size=3, stride=2, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend ([nn.ReLU(inplace=True)])
    first_conv_layer.extend(features)  
    self.mbl.features = nn.Sequential(*first_conv_layer)

    features = list(self.mbl.classifier.children())[:-3] 
    features.extend([nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)])
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)])
    features.extend([nn.AdaptiveAvgPool2d((1, 2))])
    features.extend([nn.Sigmoid()]) 
    self.mbl.classifier = nn.Sequential(*features)

  def forward(self, image):
    x = self.mbl.features(image)
    x = self.mbl.classifier(x)
    return x

#vgg16
class VGG16(nn.Module):
  def __init__(self):
    super(vgg16, self).__init__()
    self.vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)
    features = list (self.vgg16.features)[1:]
    first_conv_layer = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(features)  
    self.vgg16.features = nn.Sequential(*first_conv_layer)

    features = list(self.vgg16.classifier.children())[:-7] # Remove last layer
    features.extend([nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)])
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)])
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]) 
    features.extend([nn.AdaptiveAvgPool2d((1, 2))])
    features.extend([nn.Sigmoid()]) 
    
    self.vgg16.classifier = nn.Sequential(*features)
    self.vgg16.avgpool = None
 
  def forward(self, image):
    x = self.vgg16.features (image)
    x = self.vgg16.classifier (x)
    return x

#resnet50
class ResNet50(nn.Module):
  def __init__(self):
    super(ResNet50, self).__init__()
    self.res = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    self.res.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    
    features = list(self.res.fc.children())[:-1] 
    features.extend([nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) 
    features.extend([nn.AdaptiveAvgPool2d((1, 2))])
    features.extend([nn.Sigmoid()]) 
    self.res.avgpool = None
    self.res.fc = nn.Sequential(*features)
 
  def forward(self, image):
    x = self.res.conv1(image)
    x = self.res.bn1(x)
    x = self.res.relu(x)
    x = self.res.maxpool(x)

    x = self.res.layer1(x)
    x = self.res.layer2(x)
    x = self.res.layer3(x)
    x = self.res.layer4(x)

    x = self.res.fc(x)
    return x

#DenseNet
class DenseNet(nn.Module):
  def __init__(self):
    super(DenseNet, self).__init__()
    self.dense = torch.hub.load('pytorch/vision:v0.6.0', 'densenet169', pretrained=True)
    features = list (self.dense.features)[1:]
    first_conv_layer = [nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
    first_conv_layer.extend(features)  
    self.dense.features = nn.Sequential(*first_conv_layer)

    features = list(self.dense.classifier.children())[:-1] 
    features.extend([nn.Conv2d(1664, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)])
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)])
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)]) 
    features.extend([nn.AdaptiveAvgPool2d((1, 2))])
    features.extend([nn.Sigmoid()]) 
    
    self.dense.classifier = nn.Sequential(*features)
 
  def forward(self, image):
    x = self.dense.features (image)
    x = self.dense.classifier (x)
    return x

# #squeeze net
class SqueezeNet(nn.Module):
  def __init__(self):
    super(SqueezeNet, self).__init__()
    self.sqz = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_1', pretrained=True)
    features = list(self.sqz.features)[1:]
    first_conv_layer = [nn.Conv2d(1, 64, kernel_size=3, stride=2, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(features)  
    self.sqz.features = nn.Sequential(*first_conv_layer)

    features = list(self.sqz.classifier.children())[:-4]
    features.extend([nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]) 
    features.extend ([nn.ReLU(inplace=True)])
    features.extend([nn.MaxPool2d(2)]) 
    features.extend([nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]) 
    features.extend([nn.AdaptiveAvgPool2d((1, 2))])
    features.extend([nn.Sigmoid()]) 
    self.sqz.classifier = nn.Sequential(*features)

  def forward(self, image):
    x = self.sqz.features(image)
    x = self.sqz.classifier(x)
    return x
