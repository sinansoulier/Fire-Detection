from tqdm.auto import tqdm

import torch
import torch.nn as nn

from src.utils.hardware import device
from src.utils.object_localization import l_boxes_to_tensor

def fire_detection_loss(outputs: tuple, labels: torch.Tensor, correctness: torch.Tensor, bbox_loss: nn.Module = nn.MSELoss(), correctness_loss: nn.Module = nn.CrossEntropyLoss()) -> torch.Tensor:
    """
    Compute the loss for the fire detection model.

    Args:
        outputs (tuple): The model outputs
        labels (torch.Tensor): The ground truth bounding box labels
        correctness (torch.Tensor): The ground truth correctness labels
    Returns:
        torch.Tensor: The loss tensor of the fire detection model
    """
    y_bboxes, y_correctness = outputs
    loss = 0.0
    loss += correctness_loss(correctness, y_correctness)
    for i in range(len(y_bboxes)):
        loss += bbox_loss(y_bboxes[i], labels[i])
    return loss

def generate_correctness_labels(labels: list, num_supported_boxes: int) -> torch.Tensor:
    """
    Generate the correctness labels for the bounding boxes, given the ground truth labels.

    Args:
        labels (list): The ground truth bounding box labels
        num_supported_boxes (int): The number of supported bounding boxes
    Returns:
        torch.Tensor: A tensor of correctness labels for each bounding
    """
    correctness_labels = torch.zeros(len(labels), num_supported_boxes)
    for i in range(len(labels)):
        correctness_labels[i, :len(labels[i])] = 1
    return correctness_labels

def train_epoch(model, train_loader, val_loader, criterion, optimizer, n_supported_boxes):
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        train_loader: The DataLoader object for the training set
        criterion: The loss function
        optimizer: The optimizer
        n_supported_boxes: The number of supported bounding boxes
    Returns:
        The average training loss and validation loss
    """
    # Set the model to training mode
    model.train()
    train_loss = 0.0

    for (images, labels) in tqdm(train_loader):
        images = images.to(device())
        # Generate object presence labels for each bounding box
        correct_labels = generate_correctness_labels(labels, n_supported_boxes).to(device())
        # Convert the bounding box labels to a tensor
        labels = l_boxes_to_tensor(labels, n_supported_boxes).to(device())

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels, correct_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    # Validate the model after training iteration
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (images, labels) in tqdm(val_loader):
            images = images.to(device())
            # Generate object presence labels for each bounding box
            correct_labels = generate_correctness_labels(labels, n_supported_boxes).to(device())
            # Convert the bounding box labels to a tensor
            labels = l_boxes_to_tensor(labels, n_supported_boxes).to(device())

            outputs = model(images)
            loss = criterion(outputs, labels, correct_labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader.dataset)

    return train_loss, val_loss

def train(model, train_loader, val_loader, criterion, optimizer, n_supported_boxes, epochs=10):
    """
    Train the model for a specified number of epochs, given a set of hyperparameters and dataloaders.

    Args:
        model: The model to train
        train_loader: The DataLoader object for the training set
        criterion: The loss function
        optimizer: The optimizer
        n_supported_boxes: The number of supported bounding boxes
        epochs: The number of epochs to train the model
    """
    for epoch in range(epochs):
        train_loss, val_loss = train_epoch(model, train_loader, val_loader, criterion, optimizer, n_supported_boxes)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss} | Validation Loss: {val_loss}")