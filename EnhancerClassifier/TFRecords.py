import os
import tfrecord
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms

#serialize dataset into tfRecords
#save_Dir: directory for saving the tfRecords
#lowImgs_Dir: directory of low light images
#highImgs_Dir: directory of normal light images
#labels_Dir: directory of class labels
#recordsName: name of the tfRecords file
def serializeData (save_Dir, lowImgs_Dir, highImgs_Dir, labels_Dir, recordsName):
  writer = tfrecord.TFRecordWriter(save_Dir + "/" + recordsName)
  imgs = os.listdir (lowImgs_Dir)
  for img in imgs:
    name = img
    ###low###
    low = cv2.imread(os.path.join(lowImgs_Dir, name))
    lowImg = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)
    lowImg = lowImg [:,:,2]
    low = cv2.imencode('.png', low)[1].tostring()
    lowImg = cv2.imencode('.png', lowImg)[1].tostring()

    ###high###
    high = cv2.imread(highImgs_Dir + "/" + name) 
    highImg = cv2.cvtColor(high, cv2.COLOR_BGR2HSV)
    highImg = highImg [:,:,2]
    highImg = cv2.imencode('.png', highImg)[1].tostring()
    high = cv2.imencode('.png', high)[1].tostring()

    ###label###
    lblName = name [:name.index(".")] + ".txt"
    lblFile = open (labels_Dir + "/" + lblName)
    label = lblFile.read().split ("\n") [0].split (" ") [0]
    label = int (label)

    #writing data with keys (can be changed)
    writer.write({
        "name": (img.encode(),"byte"), 
        "lowImage": (lowImg, "byte"),
        "low": (low, "byte"),
        "highImage": (highImg, "byte"),
        "high": (high, "byte"),
        "label": (label, "int"),
    })
  writer.close()


#deserialize tfRecords
#records_Dir: directory for tfRecords file
def deSerializeData (records_Dir):
  transform=transforms.ToTensor()
  #decode the records using the serialization keys
  def decode_image(features):
      features["lowImage"] = cv2.imdecode(features ["lowImage"], -1)
      features["lowImage"] = transform (features["lowImage"])
      features["highImage"] = cv2.imdecode(features ["highImage"], -1)
      features["highImage"] = transform (features["highImage"])
      features["low"] = cv2.imdecode(features ["low"], -1)
      features["low"] = cv2.cvtColor (features["low"], cv2.COLOR_BGR2RGB)
      features["low"] = transform (features["low"])
      features["high"] = cv2.imdecode(features ["high"], -1)
      features["high"] = cv2.cvtColor (features["high"], cv2.COLOR_BGR2RGB)
      features["high"] = transform (features["high"])
      features['name'] = "".join([chr(value) for value in features['name']])
      return features

  description = {
      "name": ("byte"), 
      "lowImage": ("byte"),
      "low": ("byte"),
      "highImage": ("byte"),
      "high": ("byte"),
      "label": ("int"),
  }

  dataset = tfrecord.torch.TFRecordDataset(records_Dir,
                                         index_path=None,
                                         description=description,
                                         transform=decode_image)
  return dataset


#visualize records data including images (low or normal) and labels 
#records: tfrecords deserialized dataset
#classes: classification dataset classes (here PascalVOC)
#batchSize: batch size for displaying images
def visualizeRecords(records, classes, batchSize):
  loader = torch.utils.data.DataLoader(records, batch_size=batchSize)
  def imshow(img):
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.show()
  for batch in loader:
    data = batch 
    labels = data ['label'].reshape (1,data ['label'].size(0))[0]
    images = data ['low']
    name = data ['name'] 
    print (name)
    # show images
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batchSize)))
