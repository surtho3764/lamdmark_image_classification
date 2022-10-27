# The solution is reference here:
Paper: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf

## Data Description
Landmark recognition GLDv2 dataset is extreme imbalance class.
test dataset:In GLDv2, there are 1.6 million images belong to the 10k classes.
test dataser:In GLDv2 ,there are 200 images. 

## Solution Architecture
My solution used Sub-center ArcFace K=3, because Sub-center ArcFace a recent improvement over Sub-center ArcFace,designed to better handle noisy data.
![image] (url)


## CNN model -Ensemble 
We trained 6 models for the ensembles: EfficientNets-B7, B6, B5, B4, B3, ResNeSt-101.
It is an ensemble of 6 metric learning models. The architecture of choice is Sub-center.


## Train strategy
Training data is split into stratified 5-fold. Each model is trained on 4 folds and validated on only 1/15 of the other fold in order to save time. We used different folds to train different single models to increase model diversity.

we designed the following progressive train strategy similar to the winning solution in Google Landmark Retrieval 2020.

First,whole CNN model was used to fine-tune strategy 10 epochs with small image size 236 ,second fine-tune strategy 15-20 epochs with medium images size (512 to 768 depending on model),thrid fine-tune strategy 1-10 epoch with image size (672 to 1024) 

## Table - Ensemble model configuration 
Model          | Image Size   | Epochs | Gold | Requirement
-------------- |:-----:|-----:| ----:   |------------------------
EfficientNet-B7| 256,512,672  | 10,13,1 |    0 | Dark Age building x 2
EfficientNet-B6| 256,512,768  | 10,17,1 |  200 | Feudal Age building x 2
EfficientNet-B5| 256,576,768  | 10,16,1 |  800 | Castle Age building x 2   
EfficientNet-B4| 256,704,768  | 10,16,1 |  200 | Feudal Age building x 2
EfficientNet-B3| 256,544,1024 | 10,18,1 |  800 | Castle Age building x 2 
ResNeSt101     | 256,576,768  | 10,16,1 |  200 | Feudal Age building x 2



# Test set predicting strategy
For each image in the test set,calculate its global feature consine similarity between all data set images.
Use the top 1 nearest neighbor's class and the corresponding consine similarity as the prediction and confidence sore
for the test images.



##  Usage
1. Download Dataset GLDv2 train,test from run download_image.py and put the ./src/raw_data folder

2. Please config.yaml your configure in config fold.

3. To train a model, please run ```src/training_model.py``` with a config file as flag:


