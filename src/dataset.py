# library
# Dataset library
import cv2
import numpy as np

# image
import albumentations
import torch
from torch.utils.data import  Dataset


# train image and valil image transformation function
# 影像轉換的function，包含train image 的轉換動作和valid image 的轉換動作
def get_transforms(image_size):
    # train image transformation
    transforms_train = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ImageCompression(quality_lower=99, quality_upper=100),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
        albumentations.Normalize()
    ])

    # valil image transformation
    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val


# Landmark Dataset
# 創建讀取GLDv2資料的dataset
class Dataset_GLDv2(Dataset):
    def __init__(self, csv, mode, transform=None):

        self.csv = csv.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image_path = row['filepath']
        print("dataset imgage_path", image_path)

        # cv2(BGR), convert BGR to RGB
        image = cv2.imread(image_path)[:, :, ::-1]
        # or
        # cv_image = cv2.imread(image_path)
        # convert BGR to RGB
        # image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        # 轉換成 [channels, height, width]
        image = image.transpose(2, 0, 1)

        if self.mode == 'test':
            # return test_dats
            return torch.tensor(image)
        else:
            # return train_data ,label
            return torch.tensor(image), torch.tensor(row.landmark_id)


