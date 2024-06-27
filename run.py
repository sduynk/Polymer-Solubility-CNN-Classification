from utils import set_seed
from engine import run_trial
from data import get_transforms, CustomImageDataset, get_weighted_data_loaders, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

def run(model_constructor, root, config, use_wandb=False, checkpoint_path="./checkpoints", random_seed=0):
    
    set_seed(random_seed)

    model = model_constructor()
    train_transform, test_transform = get_transforms(config)
    train_dataset = CustomImageDataset(os.path.join(root, "train"), train_transform)
    valid_dataset = CustomImageDataset(os.path.join(root, "val"), test_transform)
    test_dataset = CustomImageDataset(os.path.join(root, "test"), test_transform)

    train_loader, valid_loader, test_loader = get_weighted_data_loaders(train_dataset, valid_dataset, test_dataset, config['batch_size'], num_workers=8, persistent_workers=True, pin_memory=True)

    valid_result, test_result = run_trial(model, config, train_loader, valid_loader, test_loader, checkpoint_path, use_wandb)

    return valid_result, test_result

def run_cross_val(model_constructor, root, config, use_wandb=False, checkpoint_path='./checkpoints/model', random_seed=0):
    """
    Run cross-validation for a given model.

    Args:
        model_constructor (function): A function that constructs the model.
        root (str): The root directory containing the data.
        config (dict): Configuration parameters for the model.
        use_wandb (bool, optional): Whether to use wandb for logging. Defaults to False.
        checkpoint_path (str, optional): Path to save the model checkpoints. Defaults to './checkpoints/model'.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        list: A list of results for each fold.
    """

    train_transform, test_transform = get_transforms(config)

    # creates stratified k folds of training data
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    # all training data
    trainval_dataset = CustomImageDataset(os.path.join(root, "trainval"), None, preload=False)

    # held out test set
    test_dataset = CustomImageDataset(os.path.join(root, "test"), test_transform, preload=False)
    
    valid_results = []
    test_results = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(trainval_dataset)), trainval_dataset.labels)):
        
        # set random seed for reproducibility
        set_seed(random_seed)

        # construct a new model for each split
        model = model_constructor()

        # Create subsets of trainval based on SKF splits
        train_dataset = Subset(trainval_dataset, train_idx, train_transform)
        valid_dataset = Subset(trainval_dataset, valid_idx, test_transform)

        # Data Preparation for each fold
        train_loader, valid_loader, test_loader = get_weighted_data_loaders(train_dataset, valid_dataset, test_dataset, config['batch_size'], num_workers=8, pin_memory=True, persistent_workers=True)
        
        # Modify checkpoint path for each fold
        fold_checkpoint_path = f"{checkpoint_path}_fold_{fold}"
        
        # Call run function for each fold
        valid_result, test_result = run_trial(model, config, train_loader, valid_loader, test_loader, fold_checkpoint_path, use_wandb)
        valid_results.append(valid_result)
        test_results.append(test_result)
    
    return valid_results, test_results
    