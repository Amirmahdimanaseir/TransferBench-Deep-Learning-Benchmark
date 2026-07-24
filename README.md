# TransferBench

TransferBench is a deep learning benchmark project that evaluates multiple transfer learning strategies on the CIFAR-10 image classification dataset using PyTorch.

The project compares three widely used convolutional and transformer-based architectures under two transfer learning settings:

- Feature Extraction
- Fine-Tuning

The objective is to provide a reproducible benchmark for comparing model accuracy, training behavior and computational cost.

---

## Models

| Model | Feature Extraction | Fine-Tuning |
|--------|-------------------|-------------|
| ResNet18 | Yes | Yes |
| EfficientNet-B0 | Yes | Yes |
| Vision Transformer (ViT-B16) | Yes | Yes |

---

## Dataset

- CIFAR-10
- 10 image classes
- Images resized to 224 × 224
- PyTorch Dataset API

---

## Project Structure

```text
TransferBench
│
├── notebooks
├── utils
├── figures
├── results
├── models
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Experiments

The repository contains eight Jupyter notebooks.

| Notebook | Description |
|----------|-------------|
| 01 | Dataset Preparation |
| 02 | ResNet18 Feature Extraction |
| 03 | ResNet18 Fine-Tuning |
| 04 | EfficientNet-B0 Feature Extraction |
| 05 | EfficientNet-B0 Fine-Tuning |
| 06 | ViT-B16 Feature Extraction |
| 07 | ViT-B16 Fine-Tuning |
| 08 | Benchmark Report |

---

## Evaluation Metrics

The following metrics are reported for every experiment:

- Accuracy
- Precision
- Recall
- F1 Score
- Training Time

---

## Generated Outputs

The project automatically generates:

- Trained Models
- Learning Curves
- Accuracy Curves
- Confusion Matrices
- Metrics CSV Files
- Benchmark Comparison
- Benchmark Report

---

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

---

## Installation

```bash
git clone https://github.com/your-username/TransferBench.git

cd TransferBench

pip install -r requirements.txt
```

---

## Running the Project

Execute the notebooks in the following order:

1. 01_Data_Preparation
2. 02_ResNet18_Feature_Extraction
3. 03_ResNet18_Fine_Tuning
4. 04_EfficientNetB0_Feature_Extraction
5. 05_EfficientNetB0_Fine_Tuning
6. 06_ViT_B16_Feature_Extraction
7. 07_ViT_B16_Fine_Tuning
8. 08_Benchmark_Report

---

## Results

The benchmark report summarizes the performance of all evaluated models and training strategies using a unified comparison table and visualization suite.

---

## License

This project is released under the MIT License.
