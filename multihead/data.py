import os
import torch
from torch.utils.data import Dataset, DataLoader
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
            s = scale*1.0/r
            return cv2.resize(img, (0, 0), fx = s, fy = s)
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

    def __init__(self, scale=300, always_apply = False, p=1.0):
        super().__init__(always_apply, p)
        self.scale = scale

    def apply(self, img, **params):
        return advanced_fundus_preprocessing(img, self.scale)

    def get_transform_init_args_names(self):
        return ("scale",)


class MultiClassSegmentationDataset(Dataset):
    def __init__(self, images_dir, hard_exudates_dir, haemorrhages_dir, transform = None):
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

        hard_exudate_mask = cv2.imread(hard_exudate_mask_path, cv2.IMREAD_GRAYSCALE)
        if hard_exudate_mask is None:
            hard_exudate_mask = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
        else:
            hard_exudate_mask = (hard_exudate_mask > 0).astype(np.uint8)

        haemorrhage_mask = cv2.imread(haemorrhage_mask_path, cv2.IMREAD_GRAYSCALE)
        if haemorrhage_mask is None:
            haemorrhage_mask = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
        else:
            haemorrhage_mask = (haemorrhage_mask > 0).astype(np.uint8)

        multi_class_mask = np.zeros_like(hard_exudate_mask)
        multi_class_mask[hard_exudate_mask > 0] = 1
        multi_class_mask[haemorrhage_mask > 0] = 2

        if self.transform:
            augmented = self.transform(image = image, mask = multi_class_mask)
            image = augmented['image']
            multi_class_mask = augmented['mask']

        return image, multi_class_mask.long()


class RetinoGradingDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform = None):
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
            augmented = self.transform(image = image)
            image = augmented['image']

        return image, torch.tensor(grade, dtype = torch.long)


def get_transforms(cfg):
    # Initialize preprocessing steps
    preprocessing_steps = []

    # Advanced fundus preprocessing
    if hasattr(cfg, 'preprocessing') and getattr(cfg.preprocessing, 'advanced_fundus', False):
        fundus_scale = getattr(cfg.preprocessing, 'fundus_scale', 300)
        preprocessing_steps.append(
            AdvancedFundusTransform(scale = fundus_scale, p = 1.0)
        )

    # CLAHE preprocessing
    if hasattr(cfg, 'preprocessing') and getattr(cfg.preprocessing, 'clahe', False):
        preprocessing_steps.append(A.CLAHE(
            clip_limit = getattr(cfg.preprocessing, 'clahe_clip_limit', 3.0),
            tile_grid_size = (getattr(cfg.preprocessing, 'clahe_tile_size', 8),
                             getattr(cfg.preprocessing, 'clahe_tile_size', 8)),
            p = 1.0
        ))

    # Center crop preprocessing
    if hasattr(cfg, 'preprocessing') and getattr(cfg.preprocessing, 'center_crop', False):
        preprocessing_steps.append(A.CenterCrop(
            height = cfg.preprocessing.crop_height,
            width = cfg.preprocessing.crop_width,
            p = 1.0
        ))

    # Base transformations for training
    train_augmentations = preprocessing_steps + [A.Resize(cfg.image_size, cfg.image_size)]

    # Add augmentations if specified in config
    if hasattr(cfg, 'augmentation'):
        # Flip augmentations
        if getattr(cfg.augmentation, 'horizontal_flip', 0) > 0:
            train_augmentations.append(A.HorizontalFlip(p = cfg.augmentation.horizontal_flip))

        if getattr(cfg.augmentation, 'vertical_flip', 0) > 0:
            train_augmentations.append(A.VerticalFlip(p = cfg.augmentation.vertical_flip))

        if getattr(cfg.augmentation, 'random_rotate90', 0) > 0:
            train_augmentations.append(A.RandomRotate90(p = cfg.augmentation.random_rotate90))

        if getattr(cfg.augmentation, 'rotation_prob', 0) > 0:
            train_augmentations.append(A.Rotate(
                limit = getattr(cfg.augmentation, 'rotation_limit', 30),
                p = cfg.augmentation.rotation_prob
            ))

        # Color augmentations
        if getattr(cfg.augmentation, 'brightness_contrast', 0) > 0:
            train_augmentations.append(A.RandomBrightnessContrast(
                brightness_limit = getattr(cfg.augmentation, 'brightness_limit', 0.3),
                contrast_limit = getattr(cfg.augmentation, 'contrast_limit', 0.3),
                p = cfg.augmentation.brightness_contrast
            ))

        if getattr(cfg.augmentation, 'hue_saturation', 0) > 0:
            train_augmentations.append(A.HueSaturationValue(
                hue_shift_limit = getattr(cfg.augmentation, 'hue_shift_limit', 25),
                sat_shift_limit = getattr(cfg.augmentation, 'sat_shift_limit', 40),
                val_shift_limit = getattr(cfg.augmentation, 'val_shift_limit', 25),
                p = cfg.augmentation.hue_saturation
            ))

        # Noise and blur augmentations
        if getattr(cfg.augmentation, 'gaussian_blur', 0) > 0:
            train_augmentations.append(A.GaussianBlur(blur_limit = (3, 7), p = cfg.augmentation.gaussian_blur))

        if getattr(cfg.augmentation, 'motion_blur', 0) > 0:
            train_augmentations.append(A.MotionBlur(blur_limit = 7, p = cfg.augmentation.motion_blur))

        if getattr(cfg.augmentation, 'gaussian_noise', 0) > 0:
            train_augmentations.append(A.GaussNoise(p = cfg.augmentation.gaussian_noise))

        if getattr(cfg.augmentation, 'elastic_transform', 0) > 0:
            train_augmentations.append(A.ElasticTransform(
                alpha = 120, sigma = 120 * 0.05,
                p = cfg.augmentation.elastic_transform
            ))

        if getattr(cfg.augmentation, 'grid_distortion', 0) > 0:
            train_augmentations.append(A.GridDistortion(p = cfg.augmentation.grid_distortion))

        if getattr(cfg.augmentation, 'optical_distortion', 0) > 0:
            train_augmentations.append(A.OpticalDistortion(
                distort_limit = 0.3,
                p = cfg.augmentation.optical_distortion
            ))
    else:
        # Default augmentations if none specified
        train_augmentations.extend([
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.5),
            A.RandomRotate90(p = 0.5),
            A.RandomBrightnessContrast(p = 0.3),
        ])

    # Final normalization and tensor conversion
    train_augmentations.extend([
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_transform = A.Compose(train_augmentations)

    # Validation transforms (no augmentation)
    val_augmentations = preprocessing_steps + [
        A.Resize(cfg.image_size, cfg.image_size),
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()
    ]

    val_transform = A.Compose(val_augmentations)

    return train_transform, val_transform


def create_dataloaders(cfg):
    train_transform, val_transform = get_transforms(cfg)

    seg_train_dataset = MultiClassSegmentationDataset(
        cfg.segmentation_data.train_images_dir,
        cfg.segmentation_data.train_hard_exudates_dir,
        cfg.segmentation_data.train_haemorrhages_dir,
        train_transform
    )
    seg_test_dataset = MultiClassSegmentationDataset(
        cfg.segmentation_data.test_images_dir,
        cfg.segmentation_data.test_hard_exudates_dir,
        cfg.segmentation_data.test_haemorrhages_dir,
        val_transform
    )

    cls_train_dataset = RetinoGradingDataset(
        cfg.classification_data.train_images_dir,
        cfg.classification_data.train_labels_csv,
        train_transform
    )
    cls_test_dataset = RetinoGradingDataset(
        cfg.classification_data.test_images_dir,
        cfg.classification_data.test_labels_csv,
        val_transform
    )

    dataloader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "pin_memory": True,
        "num_workers": cfg.training.num_workers
    }

    seg_train_loader = DataLoader(seg_train_dataset, shuffle = True, **dataloader_kwargs)
    seg_test_loader = DataLoader(seg_test_dataset, shuffle = False, **dataloader_kwargs)
    cls_train_loader = DataLoader(cls_train_dataset, shuffle = True, **dataloader_kwargs)
    cls_test_loader = DataLoader(cls_test_dataset, shuffle = False, **dataloader_kwargs)

    return seg_train_loader, seg_test_loader, cls_train_loader, cls_test_loader