import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import Dataset

def get_transforms(config):
    # ImageNet mean and std because we're using Imagenet pretrained weights
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225] 
    
    train_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.RandomResizedCrop(config['crop_size'], (config['scale_lower'], 1.0)),
        transforms.RandomHorizontalFlip(config['h_flip']),
        transforms.RandomRotation(config['rotation']),
        transforms.ColorJitter(config['brightness'], config['contrast'], config['saturation'], config['hue']),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, test_transform
    
def get_weighted_data_loaders(
    train_dataset,
    valid_dataset,
    test_dataset,
    batch_size,
    **kwargs):
    """
    Returns weighted data loaders for training, validation, and testing datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        valid_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The testing dataset.
        batch_size (int): The batch size for the data loaders.
        **kwargs: Additional keyword arguments to be passed to the DataLoader.

    Returns:
        train_loader (DataLoader): The data loader for the training dataset.
        valid_loader (DataLoader): The data loader for the validation dataset.
        test_loader (DataLoader): The data loader for the testing dataset.
    """
    
    # Weighted sampler to address class imbalance
    class_counts = np.bincount(train_dataset.labels)
    class_weights = 1./class_counts
    sample_weights = class_weights[train_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    return train_loader, valid_loader, test_loader

class CustomImageDataset(Dataset):
    """
    A custom dataset class for loading image data. This was in part generated
    by ChatGPT. Operates in a similar manner to ImageFolder but allows for
    all samples to be eagerly loaded in memory as PIL Images. Note that when
    the number of workers in a DataLoader is > 0, each worker receives a copy
    of the dataset, meaning preload = True can be memory intensive.

    Args:
        root_dir (str): The root directory of the dataset.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        preload (bool, optional): Whether to preload all images into memory. Default: False.

    Attributes:
        root_dir (str): The root directory of the dataset.
        transform (callable): A function/transform that takes in an PIL image and returns a transformed version.
        image_paths (list): A list of image file paths.
        images (list): A list of preloaded images.
        labels (list): A list of image labels.
        class_to_idx (dict): A dictionary mapping class names to class indices.
        idx_to_class (list): A list mapping class indices to class names.
        preload (bool): Whether to preload all images into memory.
    """

    def __init__(self, root_dir, transform=None, preload=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = []
        self.preload=preload
        
        self._load_data()

    def _load_data(self):
        """
        Loads image paths and labels from the root directory.
        """
        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = classes
        
        for cls in classes:
            cls_folder = os.path.join(self.root_dir, cls)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls])
        if self.preload:
            for path in self.image_paths:
                self.images.append(self._load_image(path))

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.preload:
            image = self.images[idx]
        else:
            image = self._load_image(self.image_paths[idx])

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class Subset(Dataset):
    """
    Subset of a dataset at specified indices. Adaptation of Subset
    from torch.utils.data. Allows for transforms to be defined for a
    specific subset instead of the transform used by the Superset

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.labels = (np.array(self.dataset.labels)[self.indices]).tolist()

    def __len__(self):
        if self.indices.shape == ():
            return 1
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label
