import torch
import matplotlib.pyplot as plt 
import numpy as np
import csv

def gen_config(model):
    assert model in ["efficient_net", "resnet50", "convnext_tiny"]
    if model == "efficient_net":
        config = {
            "batch_size":32,
            "epochs":50,
            "lr":0.00085,
            "weight_decay":0.008,
            "label_smoothing":0.04,
            "image_size": (256, 224),
            "crop_size": (224, 224),
            "scale_lower": 0.8,
            "rotation": 10,
            "h_flip": 0.5,
            "brightness": 0.055,
            "contrast": 0.003,
            "hue": 0.1,
            "saturation":0.075       
        }
    elif model == "resnet50":
        config = {
            "batch_size":32,
            "epochs":50,
            "lr":0.00075,
            "weight_decay":0.0005,
            "label_smoothing":0.09,
            "image_size": (256, 224),
            "crop_size": (224, 224),
            "scale_lower": 0.86,
            "rotation": 60,
            "h_flip": 0.5,
            "brightness": 0.007,
            "contrast": 0.08,
            "hue": 0.015,
            "saturation":0.03       
        }
    elif model == "convnext_tiny":
        config = {
            "batch_size":64,
            "epochs":50,
            "lr":0.000013,
            "weight_decay":0.0016,
            "label_smoothing":0.14,
            "image_size": (256, 224),
            "crop_size": (224, 224),
            "scale_lower": 0.994,
            "rotation": 60,
            "h_flip": 0.5,
            "brightness": 0.012,
            "contrast": 0.03,
            "hue": 0.15,
            "saturation":0.075       
        }

    return config

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name":"val_acc_mean"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64]},
        "epochs": {"values": [50]},
        "lr": {"max": 0.001, "min":0.00005},
        "weight_decay": {"max":0.01, "min":0.0},
        "label_smoothing": {"max":0.2, "min":0.0},
        "image_size": {"values": [(256, 256)]},
        "crop_size":  {"values": [(224, 224)]},
        "scale_lower": {"max": 1.0, "min": 0.08},
        "rotation": {"values": [10, 20, 40, 60, 100]},
        "h_flip": {"values": [0.5]},
        "brightness": {"max": 0.1, "min": 0.0},
        "contrast": {"max": 0.1, "min": 0.0},
        "hue": {"max": 0.1, "min": 0.0},
        "saturation": {"max": 0.1, "min": 0.0},
        "model": {"value": "convnext_tiny"}
    }
}

def filter_params(model):
    """
    Filters the parameters of a model into two groups: decay_parameters and no_decay_parameters.

    Parameters:
    model (torch.nn.Module): The model whose parameters will be filtered.

    Returns:
    dict: A dictionary containing two lists of parameters: decay_parameters and no_decay_parameters.
    """
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    
    return {"decay_parameters": decay, "no_decay_parameters": no_decay}


def write_dicts_to_csv(dict_list, csv_file_path, sort_key=None, ascending=True):
    if not dict_list:
        print("The list is empty.")
        return

    # Extract the keys from the first dictionary as the header
    headers = dict_list[0].keys()

    # Sort the list of dictionaries if a sort_key is provided
    if sort_key:
        dict_list = sorted(dict_list, key=lambda x: x[sort_key], reverse=not ascending)

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Write the header
        writer.writeheader()

        # Write the data rows
        for dictionary in dict_list:
            writer.writerow(dictionary)

    print(f"CSV file '{csv_file_path}' created successfully.")


def set_seed(seed: int = 42) -> None:
    import numpy as np
    import random
    import os

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set as {seed}")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def add_prefix_to_keys(d, prefix):
    """
    Adds a prefix to all keys in a dictionary.

    Parameters:
    d (dict): The dictionary to which the prefix will be added.
    prefix (str): The prefix to add to the keys.

    Returns:
    dict: A new dictionary with the prefix added to each key.
    """
    if not isinstance(d, dict):
        raise ValueError("The first argument must be a dictionary")
    if not isinstance(prefix, str):
        raise ValueError("The prefix must be a string")

    return {f"{prefix}{key}": value for key, value in d.items()}