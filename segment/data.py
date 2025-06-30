import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
Example configuration for preprocessing options:

# No preprocessing
cfg.preprocessing.clahe = False
cfg.preprocessing.center_crop = False

# Only center crop
cfg.preprocessing.clahe = False
cfg.preprocessing.center_crop = True
cfg.preprocessing.crop_height = 2848
cfg.preprocessing.crop_width = 3856

# Only CLAHE
cfg.preprocessing.clahe = True
cfg.preprocessing.clahe_clip_limit = 3.0
cfg.preprocessing.clahe_tile_size = 8
cfg.preprocessing.center_crop = False

# Both CLAHE and center crop
cfg.preprocessing.clahe = True
cfg.preprocessing.clahe_clip_limit = 3.0
cfg.preprocessing.clahe_tile_size = 8
cfg.preprocessing.center_crop = True
cfg.preprocessing.crop_height = 2848
cfg.preprocessing.crop_width = 3856

# Advanced fundus preprocessing
cfg.preprocessing.advanced_fundus = True
cfg.preprocessing.fundus_scale = 300
"""


def advanced_fundus_preprocessing(image, scale=300):
    """
    Advanced fundus preprocessing pipeline:
    1. Scale image to fixed radius
    2. Subtract local mean color using Gaussian blur
    3. Apply circular crop (remove outer 10%)
    
    This preprocessing is particularly effective for retinal images as it:
    - Standardizes image scale across dif ferent cameras/settings
    - Reduces illumination artif acts common in fundus photography
    - Focuses on diagnostically relevant areas by removing periphery
    """
def scaleRadius(img, scale):

        x= img[img.shape[0]//2,:,:].sum(1)
r=(x>x.mean()/10).sum()/2
if r>0:
            s= scale*1.0/r
return cv2.resize(img,(0,0), fx= s, fy= s)
return img

try:

        image= scaleRadius(image, scale)


gaussian_blur= cv2.GaussianBlur(image,(0,0), scale/30)
image= cv2.addWeighted(image,4, gaussian_blur,-4,128)

return image

except Exception as e:
        print(f"Warning: Advanced fundus preprocessing failed, using original image: {e}")
return image


class MultiClassDataset(Dataset):
    def __init__(self, images_dir, hard_exudates_dir, haemorrhages_dir, transform= None):
        self.images_dir= images_dir
self.hard_exudates_dir= hard_exudates_dir
self.haemorrhages_dir= haemorrhages_dir
self.transform= transform
self.images= os.listdir(images_dir)

def __len__(self):
        return len(self.images)

def __getitem__(self, idx):
        img_path= os.path.join(self.images_dir, self.images[idx])

img_name= os.path.splitext(self.images[idx])[0]
hard_exudate_mask_name= f"{img_name}_EX.tif"
haemorrhage_mask_name= f"{img_name}_HE.tif"

hard_exudate_mask_path= os.path.join(self.hard_exudates_dir, hard_exudate_mask_name)
haemorrhage_mask_path= os.path.join(self.haemorrhages_dir, haemorrhage_mask_name)

image= cv2.imread(img_path)
image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

hard_exudate_mask= cv2.imread(hard_exudate_mask_path, cv2.IMREAD_GRAYSCALE)
if hard_exudate_mask is None:
            hard_exudate_mask= np.zeros((image.shape[0], image.shape[1]), dtype= np.uint8)
else:
            hard_exudate_mask=(hard_exudate_mask>0).astype(np.uint8)

haemorrhage_mask= cv2.imread(haemorrhage_mask_path, cv2.IMREAD_GRAYSCALE)
if haemorrhage_mask is None:
            haemorrhage_mask= np.zeros((image.shape[0], image.shape[1]), dtype= np.uint8)
else:
            haemorrhage_mask=(haemorrhage_mask>0).astype(np.uint8)

multi_class_mask= np.zeros_like(hard_exudate_mask)
multi_class_mask[hard_exudate_mask>0]=1
multi_class_mask[haemorrhage_mask>0]=2

if self.transform:
            augmented= self.transform(image= image, mask= multi_class_mask)
image= augmented['image']
multi_class_mask= augmented['mask']

return image, multi_class_mask.long()


def get_transforms(cfg):

    preprocessing_steps=[]


if hasattr(cfg,'preprocessing')and getattr(cfg.preprocessing,'advanced_fundus', False):
        fundus_scale= getattr(cfg.preprocessing,'fundus_scale',300)
preprocessing_steps.append(
A.Lambda(image= lambdax,**kwargs:advanced_fundus_preprocessing(x, fundus_scale), p=1.0)
)


if hasattr(cfg,'preprocessing')andcfg.preprocessing.clahe:
        preprocessing_steps.append(A.CLAHE(
clip_limit= cfg.preprocessing.clahe_clip_limit,
tile_grid_size=(cfg.preprocessing.clahe_tile_size, cfg.preprocessing.clahe_tile_size),
p=1.0
))


if hasattr(cfg,'preprocessing')andcfg.preprocessing.center_crop:
        preprocessing_steps.append(A.CenterCrop(
height= cfg.preprocessing.crop_height,
width= cfg.preprocessing.crop_width,
p=1.0
))


train_augmentations= preprocessing_steps+[A.Resize(cfg.image_size, cfg.image_size)]


if cfg.augmentation.horizontal_flip>0:
        train_augmentations.append(A.HorizontalFlip(p= cfg.augmentation.horizontal_flip))

if cfg.augmentation.vertical_flip>0:
        train_augmentations.append(A.VerticalFlip(p= cfg.augmentation.vertical_flip))

if cfg.augmentation.random_rotate90>0:
        train_augmentations.append(A.RandomRotate90(p= cfg.augmentation.random_rotate90))

if cfg.augmentation.rotation_prob>0:
        train_augmentations.append(A.Rotate(
limit= cfg.augmentation.rotation_limit,
p= cfg.augmentation.rotation_prob
))


if cfg.augmentation.brightness_contrast>0:
        train_augmentations.append(A.RandomBrightnessContrast(
brightness_limit= cfg.augmentation.brightness_limit,
contrast_limit= cfg.augmentation.contrast_limit,
p= cfg.augmentation.brightness_contrast
))

if cfg.augmentation.hue_saturation>0:
        train_augmentations.append(A.HueSaturationValue(
hue_shift_limit= cfg.augmentation.hue_shift_limit,
sat_shift_limit= cfg.augmentation.sat_shift_limit,
val_shift_limit= cfg.augmentation.val_shift_limit,
p= cfg.augmentation.hue_saturation
))


if cfg.augmentation.gaussian_blur>0:
        train_augmentations.append(A.GaussianBlur(blur_limit=(3,7), p= cfg.augmentation.gaussian_blur))

if cfg.augmentation.motion_blur>0:
        train_augmentations.append(A.MotionBlur(blur_limit=7, p= cfg.augmentation.motion_blur))

if cfg.augmentation.gaussian_noise>0:
        train_augmentations.append(A.GaussNoise(p= cfg.augmentation.gaussian_noise))

if cfg.augmentation.elastic_transform>0:
        train_augmentations.append(A.ElasticTransform(
alpha=120, sigma=120*0.05,
p= cfg.augmentation.elastic_transform
))

if cfg.augmentation.grid_distortion>0:
        train_augmentations.append(A.GridDistortion(p= cfg.augmentation.grid_distortion))

if cfg.augmentation.optical_distortion>0:
        train_augmentations.append(A.OpticalDistortion(
distort_limit=0.3,
p= cfg.augmentation.optical_distortion
))


train_augmentations.extend([
A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
ToTensorV2()
])

train_transform= A.Compose(train_augmentations)


val_augmentations= preprocessing_steps+[
A.Resize(cfg.image_size, cfg.image_size),
A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
ToTensorV2()
]

val_transform= A.Compose(val_augmentations)

return train_transform, val_transform


def create_dataloaders(cfg):
    train_transform, val_transform= get_transforms(cfg)

train_dataset= MultiClassDataset(
cfg.data.train_images_dir,
cfg.data.train_hard_exudates_dir,
cfg.data.train_haemorrhages_dir,
train_transform
)
test_dataset= MultiClassDataset(
cfg.data.test_images_dir,
cfg.data.test_hard_exudates_dir,
cfg.data.test_haemorrhages_dir,
val_transform
)

dataloader_kwargs={
"batch_size":cfg.training.batch_size,
"pin_memory":True,
"num_workers":cfg.training.num_workers
}

if cfg.training.num_workers>0:
        dataloader_kwargs["prefetch_factor"]=2

train_loader= DataLoader(
train_dataset,
shuffle= True,
**dataloader_kwargs
)
test_loader= DataLoader(
test_dataset,
shuffle= False,
**dataloader_kwargs
)

return train_loader, test_loader