# RCH
**code** for our paper "Renovate Yourself: Calibrating Feature Representation of Misclassified Pixels for Semantic Segmentation"

### PipleLine of RCH
![image](https://github.com/VipaiLab/RCH/blob/main/images/model.png)

### Results
Network|Encoder|Iteration|Train|Test|&#929|&#932|mIoU(log)|pth|comments
:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:
OCRNet|HRNet-W48|40000|train set|val set|0.9|0.5|[82.24](https://github.com/VipaiLab/RCH/blob/main/log_files/ocr_hr48_0.5_0.9.log)|[pth](https://drive.google.com/file/d/1jlccpeFoFE7eOBx7f3463-qp9wwG5PUB/view?usp=sharing)
OCRNet|HRNet-W48|40000|train set|val set|0.9|0.5|[82.29](https://github.com/VipaiLab/RCH/blob/main/log_files/reproduce.log)|[pth](https://drive.google.com/file/d/1x0riBwzCpiFSLJrRJ6hJTSZ7hm2jm_G1/view?usp=sharing)|Reproduced
OCRNet|HRNet-W48|40000|train set|val set|0.8|0.5|[82.25](https://github.com/VipaiLab/RCH/blob/main/log_files/ocr_0.5_0.8.log)|_
OCRNet|HRNet-W48|40000|train set|val set|0.9|0.25|[81.88](https://github.com/VipaiLab/RCH/blob/main/log_files/ocr_0.25_0.9.log)|_
FCN|HRNet-W48|40000|train set|val set|0.9|0.25|[81.83](https://github.com/VipaiLab/RCH/blob/main/log_files/fcn_hr48_0.25_0.9.log)|[pth](https://drive.google.com/file/d/1UrKL69oypy9hKPBgBzUm1iSYx3jyB_Ms/view?usp=sharing)

### Requirements
python>=3.6    
torch>=1.7   
torchvision  
visdom   
numpy   
pillow   
scikit-learn

### Usage

+ core code of RCH is ./mmseg/models/decode_heads/decode_head_new.py
+ Please replace your dataset dir into the corresponding config file.
+ train OCRNet on cityscapes:
  
```./tools/dist_train.sh ./configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py 8```

Now, you can use RCH to train OCRNet, FCN, DeepLab. We will continue to update more available decoders。


### Citation
@InProceedings{wang2021rch,  
    tilte = {Renovate Yourself: Calibrating Feature Representation of Misclassified Pixels for Semantic Segmentation},  
    author = {Wang, Hualiang and Chu, Huanpeng and Fu, siming},  
    booktitle = {AAAI},  
    year = {2022}  
}


