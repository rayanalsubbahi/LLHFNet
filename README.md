# LLHFNet
Low Light Homomorphic filtering Network 

  **Requirements**: 
  
  pytorch 1.6,  pytorch-msssim, tfrecord
  
  A. 	Testing: 
  
   You can test our model using Resnet50, VGG16, MobileNetv2, DenseNet, or SqueezeNet feature extractors. You just need to specify the directory of your testing images and model      weights. 

    !python3 eval.py --device=0 --featureExt='resnet50' --test_Dir='./ /data/testImages' \
     --ckpt='./exp1_ckpt.pth' \
     --results_Dir='./results'
 
  B. Training: 
  
  You can train the model using our custom dataset based on SICE Part 1[] and using any of the extractors mentioned above. You just need to specify the directories for the low       and normal light images.
  
    !python3 train.py --device=0 --experiment='expT' --featureExt='squeezenet' \
      --base_Dir='./data/part1' --gt_Dir='./gt' \
      --ckpt_Dir='./ckpt' --numEpoch=30

