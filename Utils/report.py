"""
TransferBench
=============

Report Generator

Generate Benchmark Reports

Outputs

- CSV
- Markdown
- Ranking
"""

import os
import glob
import pandas as pd


# -----------------------------------------------------
# Load All Results
# -----------------------------------------------------

def load_all_results(results_dir="results"):

    csv_files = glob.glob(
        os.path.join(results_dir, "*_metrics.csv")
    )

    if len(csv_files) == 0:

        raise FileNotFoundError(
            "No metrics CSV files found."
        )

    dfs = []

    for file in csv_files:

        dfs.append(pd.read_csv(file))

    benchmark = pd.concat(
        dfs,
        ignore_index=True,
    )

    return benchmark


# -----------------------------------------------------
# Rank Models
# -----------------------------------------------------

def rank_models(
    benchmark,
    metric="Accuracy",
):

    return benchmark.sort_values(
        by=metric,
        ascending=False,
    ).reset_index(drop=True)


# -----------------------------------------------------
# Best Model
# -----------------------------------------------------

def best_model(
    benchmark,
    metric="Accuracy",
):

    return benchmark.loc[
        benchmark[metric].idxmax()
    ]

# -----------------------------------------------------
# Markdown Table
# -----------------------------------------------------

def dataframe_to_markdown(df):

    return df.to_markdown(
        index=False,
    )


# -----------------------------------------------------
# Save Markdown
# -----------------------------------------------------

def save_markdown(
    benchmark,
    path="results/benchmark.md",
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("# TransferBench Results\n\n")

        f.write(
            dataframe_to_markdown(
                benchmark
            )
        )

    print(f"Markdown report saved: {path}")


# -----------------------------------------------------
# Save CSV
# -----------------------------------------------------

def save_csv(
    benchmark,
    path="results/final_benchmark.csv",
):

    benchmark.to_csv(
        path,
        index=False,
    )

    print(f"CSV report saved: {path}")


# -----------------------------------------------------
# Summary
# -----------------------------------------------------

def print_summary(
    benchmark,
):

    print("=" * 70)

    print("TransferBench Benchmark")

    print("=" * 70)

    print(benchmark)

    print("=" * 70)

    best = best_model(benchmark)

    print()

    print("Best Model")

    print("-" * 70)

    print(f"Model     : {best['Model']}")
    print(f"Strategy  : {best['Strategy']}")
    print(f"Accuracy  : {best['Accuracy']:.4f}")

    print("=" * 70)


# -----------------------------------------------------
# Generate Complete Report
# -----------------------------------------------------

def generate_report():

    benchmark = load_all_results()

    benchmark = rank_models(
        benchmark,
    )

    save_csv(
        benchmark,
    )

    save_markdown(
        benchmark,
    )

    print_summary(
        benchmark,
    )

    return benchmark


# -----------------------------------------------------
# End of File
# -----------------------------------------------------