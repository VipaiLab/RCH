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
.\work_dirs\ocrnet_hr48_512x1024_40k_cityscapes_1\iter_40000.pth  

# Other logs



# model pth

# Citation
@InProceedings{wang2021rch,  
    tilte = {Renovate Yourself: Calibrating Feature Representation of Misclassified Pixels for Semantic Segmentation},  
    author = {Wang, Hualiang and Chu, Huanpeng and Fu, siming},  
    booktitle = {AAAI},  
    year = {2022}  
}


