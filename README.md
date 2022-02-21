# RCH
Code for our paper "Renovate Yourself: Calibrating Feature Representation of Misclassified Pixels for Semantic Segmentation"

![image](https://github.com/VipaiLab/RCH/blob/main/images/model.png)

# Experiment Result
![image](https://github.com/VipaiLab/RCH/blob/main/images/table.png)  

![image](https://github.com/VipaiLab/RCH/blob/main/images/sota.png)



# Requirements
python>=3.6    
torch>=1.7   
torchvision  
visdom   
numpy   
pillow   
scikit-learn

# Usage
To obtain a full-precision model, please refer train.py.   
How to train ocrnet_hr48 on cityscape   
./tools/dist_train.sh ./configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py 8 

# Sample result  
Cityscapes:  
.\work_dirs\ocrnet_hr48_512x1024_40k_cityscapes_1\20220207_215753.log 
Corresponding model pth     
[ocrnet_hr48_512x1024_40k_cityscapes](https://drive.google.com/file/d/1P6QVg-cxX44PdQbL0RiqbhtCkh6Ug1t9/view?usp=sharing)

# Other logs
./work_dirs/deeplabv3_r50-d8_512x1024_40k_cityscapes_0.1_0.125_inbatch\20210806_032608.log
./work_dirs/fcn_hr48_512x1024_40k_cityscapes/20210815_150637.log
./work_dirs/ocrnet_hr48_512x1024_40k_cityscapes_0.5_0.25/20210811_074757.log
./work_dirs/ocrnet_hr48_512x512_80k_ade20k/20210915_081129.log
./work_dirs/upernet_swin_small_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/20210911_224103.log


# model pth

# Citation
@InProceedings{wang2021rch,  
    tilte = {Renovate Yourself: Calibrating Feature Representation of Misclassified Pixels for Semantic Segmentation},  
    author = {Wang, Hualiang and Chu, Huanpeng and Fu, siming},  
    booktitle = {AAAI},  
    year = {2022}  
}


