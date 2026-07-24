"""
TransferBench
=============

Evaluation Utilities

Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Export CSV
"""

import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# -----------------------------------------------------
# Calculate Metrics
# -----------------------------------------------------

def calculate_metrics(
    targets,
    predictions,
):

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

    }


# -----------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------

def build_confusion_matrix(
    targets,
    predictions,
):

    return confusion_matrix(
        targets,
        predictions,
    )


# -----------------------------------------------------
# Print Metrics
# -----------------------------------------------------

def print_metrics(metrics):

    print("=" * 70)

    print("Evaluation Results")

    print("=" * 70)

    print(f"Accuracy  : {metrics['accuracy']:.4f}")

    print(f"Precision : {metrics['precision']:.4f}")

    print(f"Recall    : {metrics['recall']:.4f}")

    print(f"F1 Score  : {metrics['f1']:.4f}")

    print("=" * 70)

# -----------------------------------------------------
# Export Metrics
# -----------------------------------------------------

def export_metrics_csv(
    metrics,
    model_name,
    strategy,
    training_time,
    save_path,
):
    """
    Export evaluation metrics to CSV.
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    df = pd.DataFrame({
        "Model": [model_name],
        "Strategy": [strategy],
        "Accuracy": [metrics["accuracy"]],
        "Precision": [metrics["precision"]],
        "Recall": [metrics["recall"]],
        "F1 Score": [metrics["f1"]],
        "Training Time (sec)": [training_time],
    })

    df.to_csv(
        save_path,
        index=False,
    )

    return df


# -----------------------------------------------------
# Benchmark DataFrame
# -----------------------------------------------------

def create_benchmark_dataframe(*dataframes):
    """
    Merge multiple metric DataFrames into one benchmark table.
    """

    benchmark = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return benchmark


# -----------------------------------------------------
# Save Benchmark
# -----------------------------------------------------

def save_benchmark(
    benchmark_df,
    save_path="results/final_benchmark.csv",
):
    """
    Save benchmark table.
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    benchmark_df.to_csv(
        save_path,
        index=False,
    )

    print(f"Benchmark saved to: {save_path}")


# -----------------------------------------------------
# Load Benchmark
# -----------------------------------------------------

def load_metrics(path):

    return pd.read_csv(path)


# -----------------------------------------------------
# Ranking
# -----------------------------------------------------

def rank_models(
    benchmark_df,
    metric="Accuracy",
):

    return benchmark_df.sort_values(
        by=metric,
        ascending=False,
    ).reset_index(drop=True)


# -----------------------------------------------------
# Summary
# -----------------------------------------------------

def benchmark_summary(benchmark_df):

    print("=" * 70)
    print("Benchmark Summary")
    print("=" * 70)

    print(benchmark_df)

    print("=" * 70)


# -----------------------------------------------------
# End of File
# -----------------------------------------------------