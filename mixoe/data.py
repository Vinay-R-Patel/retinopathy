import os
from typing import Dict, Tuple, List

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


def advanced_fundus_preprocessing(image, scale=300):
    """
    Advanced fundus preprocessing pipeline:
    1. Scale image to fixed radius
    2. Subtract local mean color using Gaussian blur
    3. Apply circular crop (remove outer 10%)
    """
    def scaleRadius(img, scale):
        x = img[img.shape[0]//2, :, :].sum(1)
        r = (x > x.mean()/10).sum()/2
        if r > 0:
            s = scale * 1.0 / r
            return cv2.resize(img, (0, 0), fx=s, fy=s)
        return img

    try:
        image = scaleRadius(image, scale)

        gaussian_blur = cv2.GaussianBlur(image, (0, 0), scale/30)
        image = cv2.addWeighted(image, 4, gaussian_blur, -4, 128)

        return image

    except Exception as e:
        print(f"Warning: Advanced fundus preprocessing failed, using original image: {e}")
        return image


class AdvancedFundusTransform(A.ImageOnlyTransform):
    """Custom transform class for advanced fundus preprocessing that's compatible with multiprocessing"""

    def __init__(self, scale=300, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.scale = scale

    def apply(self, img, **params):
        return advanced_fundus_preprocessing(img, self.scale)

    def get_transform_init_args_names(self):
        return ("scale",)


class MultiClassSegmentationDataset(Dataset):
    def __init__(self, images_dir, hard_exudates_dir, haemorrhages_dir, transform=None):
        self.images_dir = images_dir
        self.hard_exudates_dir = hard_exudates_dir
        self.haemorrhages_dir = haemorrhages_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])

        img_name = os.path.splitext(self.images[idx])[0]
        hard_exudate_mask_name = f"{img_name}_EX.tif"
        haemorrhage_mask_name = f"{img_name}_HE.tif"

        hard_exudate_mask_path = os.path.join(self.hard_exudates_dir, hard_exudate_mask_name)
        haemorrhage_mask_path = os.path.join(self.haemorrhages_dir, haemorrhage_mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Check if hard exudate mask file exists to avoid OpenCV warnings
        if os.path.exists(hard_exudate_mask_path):
            hard_exudate_mask = cv2.imread(hard_exudate_mask_path, cv2.IMREAD_GRAYSCALE)
            if hard_exudate_mask is not None:
                hard_exudate_mask = (hard_exudate_mask > 0).astype(np.uint8)
            else:
                hard_exudate_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            hard_exudate_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Check if haemorrhage mask file exists to avoid OpenCV warnings
        if os.path.exists(haemorrhage_mask_path):
            haemorrhage_mask = cv2.imread(haemorrhage_mask_path, cv2.IMREAD_GRAYSCALE)
            if haemorrhage_mask is not None:
                haemorrhage_mask = (haemorrhage_mask > 0).astype(np.uint8)
            else:
                haemorrhage_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            haemorrhage_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        multi_class_mask = np.zeros_like(hard_exudate_mask)
        multi_class_mask[hard_exudate_mask > 0] = 1
        multi_class_mask[haemorrhage_mask > 0] = 2

        multi_class_mask = np.clip(multi_class_mask, 0, 2)

        if self.transform:
            augmented = self.transform(image=image, mask=multi_class_mask)
            image = augmented['image']
            multi_class_mask = augmented['mask']

        multi_class_mask = torch.clamp(multi_class_mask.long(), 0, 2)

        return {"image": image, "seg": multi_class_mask, "cls": torch.tensor(-1, dtype=torch.long)}


class RetinoGradingDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.data = []

        df = pd.read_csv(labels_csv)
        for _, row in df.iterrows():
            img_name = row['Image name'] + '.jpg'
            grade = int(row['Retinopathy grade'])
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                self.data.append((img_name, grade))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, grade = self.data[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {"image": image, "seg": torch.tensor(-1, dtype=torch.long), "cls": torch.tensor(grade, dtype=torch.long)}


def get_transforms(config):
    preprocessing_steps = []

    if hasattr(config, 'preprocessing') and getattr(config.preprocessing, 'advanced_fundus', False):
        fundus_scale = getattr(config.preprocessing, 'fundus_scale', 300)
        preprocessing_steps.append(
            AdvancedFundusTransform(scale=fundus_scale, p=1.0)
        )

    if hasattr(config, 'preprocessing') and getattr(config.preprocessing, 'clahe', False):
        preprocessing_steps.append(A.CLAHE(
            clip_limit=getattr(config.preprocessing, 'clahe_clip_limit', 3.0),
            tile_grid_size=(getattr(config.preprocessing, 'clahe_tile_size', 8),
                           getattr(config.preprocessing, 'clahe_tile_size', 8)),
            p=1.0
        ))

    if hasattr(config, 'preprocessing') and getattr(config.preprocessing, 'center_crop', False):
        preprocessing_steps.append(A.CenterCrop(
            height=config.preprocessing.crop_height,
            width=config.preprocessing.crop_width,
            p=1.0
        ))

    train_augmentations = preprocessing_steps + [A.Resize(config.preprocessing.image_size, config.preprocessing.image_size)]

    if hasattr(config, 'augmentations'):
        if getattr(config.augmentations, 'horizontal_flip_prob', 0) > 0:
            train_augmentations.append(A.HorizontalFlip(p=config.augmentations.horizontal_flip_prob))

        if getattr(config.augmentations, 'vertical_flip_prob', 0) > 0:
            train_augmentations.append(A.VerticalFlip(p=config.augmentations.vertical_flip_prob))

        if getattr(config.augmentations, 'rotate90_prob', 0) > 0:
            train_augmentations.append(A.RandomRotate90(p=config.augmentations.rotate90_prob))

        if getattr(config.augmentations, 'rotation_prob', 0) > 0:
            train_augmentations.append(A.Rotate(
                limit=getattr(config.augmentations, 'rotation_limit', 30),
                p=config.augmentations.rotation_prob
            ))

        if getattr(config.augmentations, 'brightness_contrast_prob', 0) > 0:
            train_augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=getattr(config.augmentations, 'brightness_limit', 0.3),
                contrast_limit=getattr(config.augmentations, 'contrast_limit', 0.3),
                p=config.augmentations.brightness_contrast_prob
            ))

        if getattr(config.augmentations, 'hue_saturation_prob', 0) > 0:
            train_augmentations.append(A.HueSaturationValue(
                hue_shift_limit=getattr(config.augmentations, 'hue_shift_limit', 25),
                sat_shift_limit=getattr(config.augmentations, 'sat_shift_limit', 40),
                val_shift_limit=getattr(config.augmentations, 'val_shift_limit', 25),
                p=config.augmentations.hue_saturation_prob
            ))

        if getattr(config.augmentations, 'color_jitter_prob', 0) > 0:
            train_augmentations.append(A.ColorJitter(p=config.augmentations.color_jitter_prob))

        if getattr(config.augmentations, 'gaussian_blur_prob', 0) > 0:
            train_augmentations.append(A.GaussianBlur(blur_limit=(3, 7), p=config.augmentations.gaussian_blur_prob))

        if getattr(config.augmentations, 'motion_blur_prob', 0) > 0:
            train_augmentations.append(A.MotionBlur(blur_limit=7, p=config.augmentations.motion_blur_prob))

        if getattr(config.augmentations, 'gaussian_noise_prob', 0) > 0:
            train_augmentations.append(A.GaussNoise(p=config.augmentations.gaussian_noise_prob))

        if getattr(config.augmentations, 'elastic_transform_prob', 0) > 0:
            train_augmentations.append(A.ElasticTransform(
                alpha=120, sigma=120*0.05,
                p=config.augmentations.elastic_transform_prob
            ))

        if getattr(config.augmentations, 'grid_distortion_prob', 0) > 0:
            train_augmentations.append(A.GridDistortion(p=config.augmentations.grid_distortion_prob))

        if getattr(config.augmentations, 'optical_distortion_prob', 0) > 0:
            train_augmentations.append(A.OpticalDistortion(
                distort_limit=0.3,
                p=config.augmentations.optical_distortion_prob
            ))
    else:
        train_augmentations.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])

    train_augmentations.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_transform = A.Compose(train_augmentations)

    val_augmentations = preprocessing_steps + [
        A.Resize(config.preprocessing.image_size, config.preprocessing.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]

    val_transform = A.Compose(val_augmentations)

    return train_transform, val_transform