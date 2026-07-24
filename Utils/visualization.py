"""
TransferBench
=============

Visualization Utilities

Generate all project figures.

Supported Plots

- Loss Curve
- Accuracy Curve
- Confusion Matrix
- Benchmark Charts
"""

import os

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# -----------------------------------------------------
# Save Figure
# -----------------------------------------------------

def save_figure(path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# -----------------------------------------------------
# Loss Curve
# -----------------------------------------------------

def plot_loss_curve(
    history,
    save_path,
):

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_loss"],
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.legend()

    save_figure(save_path)


# -----------------------------------------------------
# Accuracy Curve
# -----------------------------------------------------

def plot_accuracy_curve(
    history,
    save_path,
):

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_acc"],
        label="Train Accuracy",
        linewidth=2,
    )

    plt.plot(
        history["val_acc"],
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.legend()

    save_figure(save_path)


# -----------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------

def plot_confusion_matrix(
    matrix,
    class_names,
    save_path,
):

    fig, ax = plt.subplots(figsize=(10,10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    disp.plot(


        cmap="Blues",
        ax=ax,
        xticks_rotation=45,
    )

    plt.title("Confusion Matrix")

    save_figure(save_path)