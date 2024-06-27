import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from metrics import AverageMeter, accuracy, compute_metrics
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import add_prefix_to_keys, filter_params

def run_trial(model, config, train_loader, valid_loader, test_loader, checkpoint_path='./checkpoints/model', use_wandb=False):
    """
    Runs a training routine and reports final evaluation metrics on a test set.

    Args:
        model (torch.nn.Module): The model to train and validate.
        config (dict): A dictionary containing configuration parameters for the trial.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        valid_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        checkpoint_path (str, optional): The path to save the model checkpoints. Defaults to './checkpoints/model'.
        use_wandb (bool, optional): Whether to use wandb for logging. Defaults to False.

    Returns:
        result: The result of the hierarchical test on the trained model.
    """
    # Optimizer and Scheduler Preparation
    learning_rate = config['lr']
    weight_decay = config['weight_decay']

    params = filter_params(model)
    decay_parameters = params['decay_parameters']
    no_decay_parameters = params['no_decay_parameters']

    optimizer = AdamW([{'params': decay_parameters, 'lr': learning_rate, 'weight_decay': weight_decay},
                       {'params': no_decay_parameters, 'lr': learning_rate, 'weight_decay': 0}])
    
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=int(len(train_loader) * config['epochs']))
    scaler = torch.cuda.amp.GradScaler()

    # Callbacks
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, mode='max')

    # Run training and validation
    train(
        model, 
        optimizer, 
        scheduler, 
        scaler, 
        train_loader, 
        valid_loader, 
        config['epochs'], 
        checkpointer=checkpointer, 
        verbose=True, 
        use_wandb=use_wandb
    )
    
    # Load best model from checkpoint
    checkpointer.load_checkpoint(model)
    valid_result = hierarchical_test(model, valid_loader)
    test_result = hierarchical_test(model, test_loader)
    return valid_result, test_result

def train(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    valid_loader,
    epochs,
    device='cuda',
    verbose=True,
    label_smoothing=0.0,
    checkpointer=None,
    use_wandb=False,
    ):
    """
    Train the model for the specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        valid_loader (torch.utils.data.DataLoader): The data loader for validation data.
        epochs (int): The number of epochs to train the model.
        device (str): The device to be used for training. Default is 'cuda'.
        verbose (bool): Whether to print training progress. Default is True.
        label_smoothing (float): The label smoothing factor. Default is 0.0.
        checkpointer (ModelCheckpoint): The checkpointer object for saving the best model. Default is None.
        use_wandb (bool): Whether to log metrics to wandb. Default is False.
    """

    model.to(device)

    print("### Training ###")
    for epoch in range(epochs):

        # trains an epoch and prints some summary statistics
        train_metrics = train_epoch(model, optimizer, scheduler, scaler, train_loader, label_smoothing, device, verbose)
        val_metrics = test(model, valid_loader)

        if use_wandb:
            log("Train", train_metrics)
            log("Valid", val_metrics)

        if verbose:
            prog = int(30 * epoch / epochs)
            progress_bar = "[" + "#" * prog + "-" * (30 - prog) + "]"
            print(("\nEpoch {}: " + progress_bar + "\n").format(epoch))
            print_summary("Train", train_metrics)
            print("\n### Validation and Checkpointing ###")
            print_summary("Validation", val_metrics)
            print("\n")

        if checkpointer is not None:
            checkpointer.save_checkpoint(model, val_metrics['accuracy'])
                  
def train_epoch(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    label_smoothing=0.0,
    device='cuda',
    verbose=True,
):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        label_smoothing (float): The label smoothing factor. Default is 0.0.
        device (str): The device to be used for training. Default is 'cuda'.
        verbose (bool): Whether to print training progress. Default is True.

    Returns:
        dict: A dictionary containing the average loss and accuracy for the epoch.
    """

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.train()

    for i, (images, labels) in enumerate(train_loader):

        # with autocast for mixed precision
        with torch.autocast(device_type=device, dtype=torch.float16):

            # forward
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = F.cross_entropy(predictions, labels, label_smoothing=label_smoothing)
        
        # backwards with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        # note that because of the scaler, the optimizer isn't always stepped on the first iteration
        # this gives a warning that the scheduler is stepped before the optimizer
        # This doesn't appear to be problematic
        scheduler.step()

        # check loss is not nan
        assert not torch.isnan(loss), "loss is nan"

        # basic metrics
        acc = accuracy(predictions, labels)
        loss_meter.update(loss.item(), n=images.shape[0])
        acc_meter.update(acc, n=images.shape[0])

        if verbose:
            msg = "Batch: {batch} - Loss: {loss:.2f} --- Accuracy: {accuracy:.2f}"
            print(msg.format(batch=i, loss=loss_meter.val, accuracy=acc_meter.val))

    return {'loss': loss_meter.avg, 'acc': acc_meter.avg}

def predict(
    model,
    dataloader,
    device='cuda'
    ):
    
    model.eval()

    y_pred = []
    y_true = []
    y_loss = []
    
    with torch.no_grad():
        
        for i, (images, labels) in enumerate(dataloader):

            with torch.autocast(device_type=device, dtype=torch.float16):
            
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                loss = F.cross_entropy(predictions, labels, reduction='none')
                
                # metrics
                y_loss.append(loss)
                y_pred.append(torch.argmax(predictions, dim=-1))
                y_true.append(labels)
    
    y_pred = torch.cat(y_pred, dim=0).tolist()
    y_true = torch.cat(y_true, dim=0).tolist()
    y_loss = torch.cat(y_loss, dim=0).mean().item()
    return {"y_true": y_true, "y_pred": y_pred, "y_loss": y_loss}


def test(model, test_loader, idx_to_class=None):
    """
    Evaluate the performance of the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        idx_to_class (dict, optional): A dictionary mapping class indices to class labels. 
            If provided, the predicted class labels will be converted from indices to labels.
            Mainly serves the purpose of implementing heirarchical testing. 
            See hierarchical_test()

    Returns:
        dict: A dictionary containing the loss and evaluation metrics.
    """

    model.eval()
    
    out = predict(model, test_loader)
    y_true, y_pred, y_loss = out['y_true'], out["y_pred"], out['y_loss']
    
    if idx_to_class is not None:
        y_true = [idx_to_class[idx] for idx in y_true]
        y_pred = [idx_to_class[idx] for idx in y_pred]

    metrics = compute_metrics(y_true, y_pred)

    return {"loss": y_loss} | metrics

def hierarchical_test(model, test_loader):
    """
    Perform hierarchical testing on the given model using the provided test loader.
    This is similar to test, but computes metrics at different class hierarchy levels.
    e.g. the soluble class can be split into colloidal and truly soluble subclasses

    Args:
        model: The model to be tested.
        test_loader: The data loader for the test dataset.

    Returns:
        A dictionary containing the test results.
    """
    class_4 = test(model, test_loader, ["colloidal", "insoluble", "partially_soluble", "soluble"])
    class_3 = test(model, test_loader, ["colloidal", "insoluble", "insoluble", "soluble"])
    class_2 = test(model, test_loader, ["soluble", "insoluble", "insoluble", "soluble"])

    class_2 = add_prefix_to_keys(class_2, "2_class_")
    class_3 = add_prefix_to_keys(class_3, "3_class_")
    class_4 = add_prefix_to_keys(class_4, "4_class_")

    all_results = class_2 | class_3 | class_4

    return all_results


###### --- Some Extra Utilities for Training and Logging --- ######

class ModelCheckpoint:
    """
    A class for saving and loading model checkpoints based on validation accuracy.

    Args:
        filepath (str): The file path to save the checkpoint.
        mode (str, optional): The mode for comparing validation accuracy. Should be either 'min' or 'max'. 
            Defaults to 'max'.

    Raises:
        ValueError: If the mode is not 'min' or 'max'.

    Attributes:
        filepath (str): The file path to save the checkpoint.
        best_accuracy (float): The best validation accuracy achieved so far.
        mode (str): The mode for comparing validation accuracy.

    Methods:
        save_checkpoint: Saves the model checkpoint if the validation accuracy is better than the previous best.
        load_checkpoint: Loads the model checkpoint from the specified file path.

    """

    def __init__(self, filepath, mode='max'):
        self.filepath = filepath
        self.best_accuracy = None
        self.mode = mode
        if mode not in ['min', 'max']:
            raise ValueError("Mode should be either 'min' or 'max'.")

    def save_checkpoint(self, model, validation_accuracy):
        """
        Saves the model checkpoint if the validation accuracy is better than the previous best.

        Args:
            model: The model to save.
            validation_accuracy (float): The validation accuracy of the model.

        """
        if self.best_accuracy is None or \
           (self.mode == 'max' and validation_accuracy > self.best_accuracy) or \
           (self.mode == 'min' and validation_accuracy < self.best_accuracy):
            
            self.best_accuracy = validation_accuracy
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_accuracy': self.best_accuracy
            }
            
            torch.save(checkpoint, self.filepath)
            print(f"Checkpoint saved with validation accuracy: {validation_accuracy:.4f}")

    def load_checkpoint(self, model):
        """
        Loads the model checkpoint from the specified file path.

        Args:
            model: The model to load the checkpoint into.

        """
        checkpoint = torch.load(self.filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        print(f"Checkpoint loaded with validation accuracy: {self.best_accuracy:.4f}")

def print_summary(tag, metrics):
    result = ' --- '.join([tag + "_" + f"{key}: {value:.4f}" for key, value in metrics.items()])
    print(result)

def log(tag, metrics):
    log_dict = {tag +"_"+ key: value for key, value in metrics.items()}
    wandb.log(log_dict)
    