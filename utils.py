import os
import shutil
import sys

import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from model import *

def refine_image(img, filterHF, device):
    batch = img.size (0)
    w,h = img.size (2), img.size (3)
    yL = filterHF [:,0]
    yH = filterHF [:,1]
    im = torch.log (img + 0.00000001)
    fft = torch.rfft(im, signal_ndim=2, normalized=False, onesided=False)
    H = torch.ones(batch,1,w,h,2).to(device)
    for i in range (batch):
      H [i,:,:,:,:] *= yH [i]
      H [i,:,0,0,:] = yL [i] 
    S = fft*H
    s = torch.irfft(S, signal_ndim=2, normalized=False, onesided=False)
    enh = torch.exp(s)
    imin = torch.Tensor(1).to(device).fill_(0)
    imax = torch.Tensor(1).to(device).fill_(1)
    enh = torch.max(imin,torch.min(imax,enh))
    return enh

def mssim (enhances,normals):
  return 1-ms_ssim(enhances, normals, data_range=1, size_average=True)


def save_ckpt(state, is_best, experiment, epoch, ckpt_dir):
    filename = os.path.join(ckpt_dir, f'{experiment}_ckpt.pth')
    torch.save(state, filename)
    if is_best:
        print(f'[BEST MODEL] Saving best model, obtained on epoch = {epoch + 1}')
        shutil.copy(filename, os.path.join(ckpt_dir, f'{experiment}_best_model.pth'))

def group_params(model):
    decay, no_decay = [], []
    for name, par in model.named_parameters():
        if 'bias' in name:
            no_decay.append(par)
        else:
            decay.append(par)

    groups = [dict(params=decay), dict(params=no_decay, weight_decay=0.)]
    return groups

def switch_model(model):
    if model == 'resnet50':
      return  ResNet50()
    elif model == 'mobilenetv2': 
      return MobileNetv2()
    elif model == 'vgg16': 
      return VGG16()
    elif model == 'densenet': 
      return DenseNet()
    elif model == 'squeezenet':  
      return SqueezeNet()
  
