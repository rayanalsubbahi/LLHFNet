# LLHFNet
Low Light Homomorphic filtering Network 

  **Requirements**: 
  
  pytorch 1.6,  pytorch-msssim, tfrecord
  
  A. 	**Testing:**
  
   You can test our model using [Resnet50](), [VGG16](), [MobileNetv2](), [DenseNet](), or [SqueezeNet]() feature extractors (weights through hyperlinks). You just need to specify the directory of your testing images and model weights. 

    !python3 eval.py --device=0 --featureExt='resnet50' --test_Dir= './data/part2' \
      --ckpt_Dir='./exp1_ckpt.pth' --results_Dir='./res'
 
  B. **Training:** 
  
  You can train the model using our custom dataset based on [SICE Part 1](), and using any of the extractors mentioned above. You just need to specify the directories for the low and normal light images.
  
    !python3 train.py --device=0 --experiment='expT' --featureExt='squeezenet' \
      --base_Dir='./data/part1' --gt_Dir='./gt' \
      --ckpt_Dir='./ckpt' --numEpoch=30

# Enhancer Classifer
  A. 	**Testing:**
  
  You need to specify the directory of the test images and model weights. You can find our custom [Pascal VOC]() and [ExDark]() evaluation datasets through the hyperlinks. 


    !python3 eval.py --device=0  --test_Dir='./data' \
      --ckpt_Dir='./exp1_ckpt.pth'  --results_Dir='./results'
      
   The output enhanced images are saved in the results directory and the code prints the image name and the corresponding predicted class label in the console. 
      
  B. **Training:** 

  You can train the model using our [custom dataset]() based on Pascl VOC. 
  Our dataset is serialized into tfRecords. You can find in TFRecords.py the codes for serializing and deserializing the dataset. 

    !python3 train.py --device=0 --experiment='expT' \
      --base_Dir='./EnhCls' --ckpt_Dir='./ckpt' --numEpoch=30

      
