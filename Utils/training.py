"""
TransferBench
=============

Training Utilities

This module contains reusable training functions
for all benchmark experiments.

Supported Models

- ResNet18
- EfficientNet-B0
- ViT-B16
"""

import os
import time
import torch


# -----------------------------------------------------
# Average Meter
# -----------------------------------------------------

class AverageMeter:
    """
    Computes and stores the average value.
    """

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


# -----------------------------------------------------
# Accuracy
# -----------------------------------------------------

def calculate_accuracy(outputs, labels):

    _, predicted = outputs.max(1)

    correct = predicted.eq(labels).sum().item()

    return correct / labels.size(0)


# -----------------------------------------------------
# Train One Epoch
# -----------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        acc = calculate_accuracy(outputs, labels)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return loss_meter.avg, acc_meter.avg


# -----------------------------------------------------
# Validation
# -----------------------------------------------------

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            acc = calculate_accuracy(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

            _, predicted = outputs.max(1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return (
        loss_meter.avg,
        acc_meter.avg,
        predictions,
        targets,
    )


# -----------------------------------------------------
# Save Checkpoint
# -----------------------------------------------------

def save_checkpoint(
    model,
    path,
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        path,
    )


# -----------------------------------------------------
# Load Checkpoint
# -----------------------------------------------------

def load_checkpoint(
    model,
    path,
    device,
):

    model.load_state_dict(

        torch.load(
            path,
            map_location=device,
        )

    )

    return model

# -----------------------------------------------------
# Train Model
# -----------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device,
    checkpoint_path,
):

    history = {

        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],

    }

    best_accuracy = 0.0

    start_time = time.time()

    print("=" * 70)
    print("Training Started")
    print("=" * 70)

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, predictions, targets = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            save_checkpoint(
                model,
                checkpoint_path,
            )

    total_time = time.time() - start_time

    print("=" * 70)
    print("Training Finished")
    print("=" * 70)

    print(f"Best Validation Accuracy : {best_accuracy:.4f}")
    print(f"Training Time            : {total_time:.2f} sec")

    return {

        "history": history,

        "best_accuracy": best_accuracy,

        "training_time": total_time,

        "predictions": predictions,

        "targets": targets,

        "checkpoint": checkpoint_path,

    }


# -----------------------------------------------------
# Print Training Summary
# -----------------------------------------------------

def print_summary(results):

    print("\n")

    print("=" * 70)

    print("Training Summary")

    print("=" * 70)

    print(f"Best Accuracy : {results['best_accuracy']:.4f}")

    print(f"Training Time : {results['training_time']:.2f} sec")

    print(f"Checkpoint    : {results['checkpoint']}")

    print("=" * 70)


# -----------------------------------------------------
# End of File
# -----------------------------------------------------