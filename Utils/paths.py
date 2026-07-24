"""
TransferBench
=============

Project Paths

Centralized path management for the entire project.
"""

from pathlib import Path


# -----------------------------------------------------
# Project Root
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------
# Directories
# -----------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

MODEL_DIR = PROJECT_ROOT / "models"

RESULT_DIR = PROJECT_ROOT / "results"

FIGURE_DIR = PROJECT_ROOT / "figures"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

UTILS_DIR = PROJECT_ROOT / "utils"


# -----------------------------------------------------
# Create Directories Automatically
# -----------------------------------------------------

for directory in [

    DATA_DIR,

    MODEL_DIR,

    RESULT_DIR,

    FIGURE_DIR,

]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------

def get_model_path(model_name):

    return MODEL_DIR / f"{model_name}.pth"


def get_metrics_path(model_name):

    return RESULT_DIR / f"{model_name}_metrics.csv"


def get_loss_plot_path(model_name):

    return FIGURE_DIR / f"{model_name}_loss.png"


def get_accuracy_plot_path(model_name):

    return FIGURE_DIR / f"{model_name}_accuracy.png"


def get_confusion_matrix_path(model_name):

    return FIGURE_DIR / f"confusion_matrix_{model_name}.png"


def get_report_path():

    return RESULT_DIR / "benchmark.md"


def get_benchmark_csv():

    return RESULT_DIR / "final_benchmark.csv"