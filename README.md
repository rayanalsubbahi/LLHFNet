# LLHFNet
Low Light Homomorphic filtering Network 

  **Requirements**: 
  
  pytorch 1.6,  pytorch-msssim, tfrecord
  
  A. 	**Testing:**
  
   You can test our model using [Resnet50](https://drive.google.com/file/d/1xFOIHuhIIjq14srgmQGfexl1d2ilHOxw/view?usp=sharing), [VGG16](https://drive.google.com/file/d/1RyBJbpSEw-WQhlpQMhvFtoC8_1kXiL32/view?usp=sharing), [MobileNetv2](https://drive.google.com/file/d/15D_zp42yUfhqOO1VFiNTZGhstJr29uda/view?usp=sharing), [DenseNet](https://drive.google.com/file/d/1-5TYQhn7cpdZsjplYhwdKGKr_bq7VQ-1/view?usp=sharing), or [SqueezeNet](https://drive.google.com/file/d/1-Ae37WXvLRF2ZBAVbVNJpC7hecTkcgQo/view?usp=sharing) feature extractors (weights through hyperlinks). You just need to specify the directory of your testing images and model weights. 

    !python3 eval.py --device=0 --featureExt='resnet50' --test_Dir= './data/part2' \
      --ckpt_Dir='./exp1_ckpt.pth' --results_Dir='./res'
 
  B. **Training:** 
  
  You can train the model using our [custom dataset](https://drive.google.com/drive/folders/1hMXkSCr7kj9AY22DkWbhJ7GJUa6RwApO?usp=sharing) based on SICE Part 1, and using any of the extractors mentioned above. You just need to specify the directories for the low and normal light images.
  
    !python3 train.py --device=0 --experiment='expT' --featureExt='squeezenet' \
      --base_Dir='./data/part1' --gt_Dir='./gt' \
      --ckpt_Dir='./ckpt' --numEpoch=30

# Enhancer Classifer
  A. 	**Testing:**
  
  You need to specify the directory of the test images and [model weights](https://drive.google.com/file/d/1-MbNOd2pkz8l6HeFVWQIPyvGTLxazaAF/view?usp=sharing). You can find our custom [Pascal VOC](https://drive.google.com/drive/folders/1HqXNWhmewNQ8WJp4gzVLjJL7YHQt4nP6?usp=sharing) and [ExDark](https://drive.google.com/drive/folders/17kKu5ci0ifIFmSdv9_myWRveynigtWvV?usp=sharing) evaluation datasets through the hyperlinks. 


    !python3 eval.py --device=0  --test_Dir='./data' \
      --ckpt_Dir='./exp1_ckpt.pth'  --results_Dir='./results'
      
   The output enhanced images are saved in the results directory and the code prints the image name and the corresponding predicted class label in the console. 
      
  B. **Training:** 

  You can train the model using our [custom dataset](https://drive.google.com/drive/folders/1SSPyP5-aY9fav6SS6KnknfCa1YsV2rMG?usp=sharing) based on Pascl VOC. 
  Our dataset is serialized into tfRecords. You can find in TFRecords.py the codes for serializing and deserializing the dataset. 

    !python3 train.py --device=0 --experiment='expT' \
      --base_Dir='./EnhCls' --ckpt_Dir='./ckpt' --numEpoch=30

      
