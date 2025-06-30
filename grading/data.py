import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


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

        return image, torch.tensor(grade, dtype=torch.long)


def advanced_fundus_preprocessing(image, scale=300):
    """
    Advanced fundus preprocessing pipeline:
    1. Scale image to fixed radius
    2. Subtract local mean color
    3. Apply circular crop (remove outer 10%)
    """
    def scaleRadius(img, scale):
        x = img[img.shape[0]//2, :, :].sum(1)
        r = (x > x.mean()/10).sum()/2
        if r > 0:
            s = scale*1.0/r
            return cv2.resize(img, (0, 0), fx=s, fy=s)
        return img

    try:
        image = scaleRadius(image, scale)

        gaussian_blur = cv2.GaussianBlur(image, (0, 0), scale/30)
        image = cv2.addWeighted(image, 4, gaussian_blur, -4, 128)

        return image

    except Exception as e:
        print(f"Warning: Advanced preprocessing failed, using original image: {e}")
        return image


def get_transforms(cfg):
    preprocessing_steps = []

    if hasattr(cfg, 'preprocessing') and getattr(cfg.preprocessing, 'advanced_fundus', False):
        fundus_scale = getattr(cfg.preprocessing, 'fundus_scale', 300)
        preprocessing_steps.append(
            A.Lambda(image=lambda x, **kwargs: advanced_fundus_preprocessing(x, fundus_scale), p=1.0)
        )

    if hasattr(cfg, 'preprocessing') and cfg.preprocessing.clahe:
        clahe_clip_limit = getattr(cfg.preprocessing, 'clahe_clip_limit', 3.0)
        clahe_tile_size = getattr(cfg.preprocessing, 'clahe_tile_size', 8)
        preprocessing_steps.append(A.CLAHE(
            clip_limit=clahe_clip_limit,
            tile_grid_size=(clahe_tile_size, clahe_tile_size),
            p=1.0
        ))

    if hasattr(cfg, 'preprocessing') and cfg.preprocessing.center_crop:
        crop_height = getattr(cfg.preprocessing, 'crop_height', 2848)
        crop_width = getattr(cfg.preprocessing, 'crop_width', 3856)
        preprocessing_steps.append(A.CenterCrop(
            height=crop_height,
            width=crop_width,
            p=1.0
        ))

    train_augmentations = preprocessing_steps + [A.Resize(cfg.image_size, cfg.image_size)]

    if hasattr(cfg, 'augmentation'):
        if cfg.augmentation.horizontal_flip > 0:
            train_augmentations.append(A.HorizontalFlip(p=cfg.augmentation.horizontal_flip))

        if cfg.augmentation.vertical_flip > 0:
            train_augmentations.append(A.VerticalFlip(p=cfg.augmentation.vertical_flip))

        if cfg.augmentation.random_rotate90 > 0:
            train_augmentations.append(A.RandomRotate90(p=cfg.augmentation.random_rotate90))

        if cfg.augmentation.rotation_prob > 0:
            train_augmentations.append(A.Rotate(
                limit=cfg.augmentation.rotation_limit,
                p=cfg.augmentation.rotation_prob
            ))

        if cfg.augmentation.brightness_contrast > 0:
            train_augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=cfg.augmentation.brightness_limit,
                contrast_limit=cfg.augmentation.contrast_limit,
                p=cfg.augmentation.brightness_contrast
            ))

        if cfg.augmentation.hue_saturation > 0:
            train_augmentations.append(A.HueSaturationValue(
                hue_shift_limit=cfg.augmentation.hue_shift_limit,
                sat_shift_limit=cfg.augmentation.sat_shift_limit,
                val_shift_limit=cfg.augmentation.val_shift_limit,
                p=cfg.augmentation.hue_saturation
            ))

        if cfg.augmentation.gaussian_blur > 0:
            train_augmentations.append(A.GaussianBlur(blur_limit=(3, 7), p=cfg.augmentation.gaussian_blur))

        if cfg.augmentation.motion_blur > 0:
            train_augmentations.append(A.MotionBlur(blur_limit=7, p=cfg.augmentation.motion_blur))

        if cfg.augmentation.gaussian_noise > 0:
            train_augmentations.append(A.GaussNoise(p=cfg.augmentation.gaussian_noise))

        if cfg.augmentation.elastic_transform > 0:
            train_augmentations.append(A.ElasticTransform(
                alpha=120, sigma=120*0.05,
                p=cfg.augmentation.elastic_transform
            ))

        if cfg.augmentation.grid_distortion > 0:
            train_augmentations.append(A.GridDistortion(p=cfg.augmentation.grid_distortion))

        if cfg.augmentation.optical_distortion > 0:
            train_augmentations.append(A.OpticalDistortion(
                distort_limit=0.3,
                p=cfg.augmentation.optical_distortion
            ))
    elif hasattr(cfg, 'augmentations'):
        augmentations_config = cfg.augmentations.augmentations if hasattr(cfg.augmentations, 'augmentations') else cfg.augmentations
        if augmentations_config.name == "light_geometric":
            train_augmentations.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ])
        elif augmentations_config.name == "color_changing":
            train_augmentations.extend([
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.ColorJitter(p=0.3)
            ])
        elif augmentations_config.name == "heavy":
            train_augmentations.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.ColorJitter(p=0.3),
                A.ShiftScaleRotate(p=0.3),
                A.RandomGamma(p=0.2),
                A.ElasticTransform(p=0.2)
            ])

    train_augmentations.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_transform = A.Compose(train_augmentations)

    val_augmentations = preprocessing_steps + [
        A.Resize(cfg.image_size, cfg.image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]

    val_transform = A.Compose(val_augmentations)

    return train_transform, val_transform


def create_dataloaders(cfg):
    train_transform, val_transform = get_transforms(cfg)

    train_dataset = RetinoGradingDataset(
        cfg.data.train_images_dir,
        cfg.data.train_labels_csv,
        train_transform
    )
    test_dataset = RetinoGradingDataset(
        cfg.data.test_images_dir,
        cfg.data.test_labels_csv,
        val_transform
    )

    dataloader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "pin_memory": True,
        "num_workers": cfg.training.num_workers
    }

    if cfg.training.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    return train_loader, test_loader