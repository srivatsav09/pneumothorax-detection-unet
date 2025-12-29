"""
Multi-Task Data Generator for Pneumothorax Classification + Localization
Generates batches with both classification labels and bounding box coordinates
"""

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from keras.utils import Sequence
from preprocessing import preprocess_dicom_for_training, create_bounding_box_from_mask


class MultitaskGenerator(Sequence):
    """
    Generator that returns batches of images with:
    1. Classification labels (0 or 1)
    2. Bounding box coordinates [x_min, y_min, x_max, y_max] (normalized 0-1)

    For negative samples (no pneumothorax), bbox is set to [0, 0, 0, 0] with zero weight.
    """

    def __init__(self, dataframe, image_path, batch_size, resize_to=512, shuffle=True,
                 rotate=8, horizontal_flip=0.5, zoom=0.15, brightness=10, contrast=0.2,
                 shear=6, aug=True):
        """
        Args:
            dataframe: pandas DataFrame with columns ['ImageId', 'EncodedPixels']
            image_path: Path to DICOM images directory
            batch_size: Number of samples per batch
            resize_to: Target image size
            shuffle: Whether to shuffle data each epoch
            rotate, horizontal_flip, zoom, brightness, contrast, shear: Augmentation parameters
            aug: Whether to apply augmentation
        """
        # Data parameters
        self.df = dataframe
        self.image_path = image_path
        self.image_filenames = self.df.index.to_list()
        self.index = np.arange(len(self.image_filenames))

        # Model parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize_to = resize_to

        # Augmentation parameters
        self.aug = aug
        if self.aug:
            self.seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-rotate, rotate),
                    shear=(-shear, shear),
                    scale=(1 - zoom, 1 + zoom)
                ),
                iaa.Fliplr(horizontal_flip),
                iaa.Add((-brightness, brightness)),  # Brightness
                iaa.LinearContrast((1 - contrast, 1 + contrast))  # Contrast
            ])

        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Returns:
            X: Image batch (batch_size, resize_to, resize_to, 3)
            [y_class, y_bbox]: Classification labels and bounding boxes
        """
        # Generate batch indices
        batch_indices = self.index[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y_class, y_bbox = self.__data_generation(batch_indices)

        return X, [y_class, y_bbox]

    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.index)

    def __data_generation(self, batch_indices):
        """
        Generate batch of images and labels.

        Args:
            batch_indices: Indices of samples in this batch

        Returns:
            X: Image batch (batch_size, resize_to, resize_to, 3)
            y_class: Classification labels (batch_size, 1)
            y_bbox: Bounding boxes (batch_size, 4)
        """
        X = np.empty((self.batch_size, self.resize_to, self.resize_to, 3), dtype=np.float32)
        y_class = np.empty((self.batch_size, 1), dtype=np.float32)
        y_bbox = np.empty((self.batch_size, 4), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            filename = self.image_filenames[idx]
            rle_mask = self.df.loc[filename, 'EncodedPixels']

            # Load and preprocess image
            image = self.__load_image(filename)

            # Determine classification label
            has_pneumothorax = (rle_mask != ' -1' and rle_mask != '-1')
            y_class[i, 0] = 1.0 if has_pneumothorax else 0.0

            # Get bounding box (or zeros for negative samples)
            if has_pneumothorax:
                bbox = create_bounding_box_from_mask(rle_mask, 1024, 1024)
                if bbox is not None:
                    y_bbox[i] = bbox
                else:
                    # Mask exists but couldn't create bbox (edge case)
                    y_bbox[i] = [0.0, 0.0, 0.0, 0.0]
                    y_class[i, 0] = 0.0  # Treat as negative
            else:
                y_bbox[i] = [0.0, 0.0, 0.0, 0.0]

            # Convert to RGB and normalize
            rgb_image = np.stack([image, image, image], axis=-1)

            # Apply augmentation
            if self.aug:
                rgb_image = self.seq(image=rgb_image)

            # Normalize to 0-1
            X[i] = rgb_image.astype(np.float32) / 255.0

        return X, y_class, y_bbox

    def __load_image(self, filename):
        """
        Load and preprocess a DICOM image using improved preprocessing.

        Args:
            filename: Image filename (without .dcm extension)

        Returns:
            Preprocessed grayscale image (resize_to, resize_to)
        """
        dicom_path = f"{self.image_path}{filename}.dcm"

        try:
            image = preprocess_dicom_for_training(
                dicom_path,
                resize_to=self.resize_to,
                apply_windowing=True,
                remove_borders=True,
                center_crop=True
            )
            return image
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            # Return blank image on error
            return np.zeros((self.resize_to, self.resize_to), dtype=np.uint8)


def create_balanced_dataset(dataframe, positive_ratio=0.5):
    """
    Create a balanced dataset by undersampling the majority class.

    Args:
        dataframe: Original DataFrame
        positive_ratio: Desired ratio of positive samples (default: 0.5 for 50/50 split)

    Returns:
        Balanced DataFrame
    """
    positive_samples = dataframe[dataframe['EncodedPixels'] != ' -1']
    negative_samples = dataframe[dataframe['EncodedPixels'] == ' -1']

    n_positive = len(positive_samples)
    n_negative_target = int(n_positive * (1 - positive_ratio) / positive_ratio)

    # Undersample negative class
    if n_negative_target < len(negative_samples):
        negative_samples = negative_samples.sample(n=n_negative_target, random_state=42)

    # Combine and shuffle
    balanced_df = pd.concat([positive_samples, negative_samples])
    balanced_df = balanced_df.sample(frac=1, random_state=42)  # Keep the ImageId index!

    print(f"Balanced dataset: {len(positive_samples)} positive, {len(negative_samples)} negative")
    print(f"Total: {len(balanced_df)} samples ({len(positive_samples)/len(balanced_df)*100:.1f}% positive)")

    return balanced_df
