from sklearn.metrics import confusion_matrix
import numpy as np
import torch

class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(predictions, targets):
    predictions = torch.argmax(predictions, dim=-1)
    acc = torch.eq(predictions, targets).float().mean()
    return acc * 100

def compute_metrics(y_true, y_pred):
    """
    Compute various evaluation metrics based on the true labels and predicted labels.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Returns:
    - metrics (dict): Dictionary containing the computed metrics.

    """

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Compute precision for each class
    precision = np.diag(cm) / np.sum(cm, axis=0)

    # Compute recall for each class
    recall = np.diag(cm) / np.sum(cm, axis=1)

    # Compute overall accuracy
    accuracy = np.diag(cm).sum() / np.sum(cm)

    # Return the computed metrics as a dictionary, macro averages computed for precision and recall
    return {"accuracy": accuracy, "mean_precision": precision.mean(), "mean_recall": recall.mean()}