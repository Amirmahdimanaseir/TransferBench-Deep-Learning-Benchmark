
# TransferBench Benchmark Report

## Best Performing Model

Model: ViT-B16

Strategy: Fine-Tuning

Accuracy: 0.9693

Precision: 0.9694

Recall: 0.9693

F1 Score: 0.9693

Training Time: 11270.48 sec

---

## Complete Benchmark

| Model           | Strategy                 |   Accuracy |   Precision |   Recall |   F1 Score |   Training Time (sec) | Experiment                    |         F1 |
|:----------------|:-------------------------|-----------:|------------:|---------:|-----------:|----------------------:|:------------------------------|-----------:|
| EfficientNet-B0 | Fine-Tuning              |     0.8905 |    0.891752 |   0.8905 |   0.890035 |               1525.12 | efficientnet_finetune_metrics | nan        |
| EfficientNet-B0 | Feature Extraction       |     0.8354 |    0.835958 |   0.8354 |   0.834041 |               3464.28 | efficientnet_freeze_metrics   | nan        |
| ResNet18        | Fine-Tuning              |     0.9399 |    0.940089 |   0.9399 |   0.939756 |               1634.35 | resnet18_finetune_metrics     | nan        |
| ResNet18        | Freeze Feature Extractor |     0.7887 |    0.794424 |   0.7887 | nan        |                435.57 | resnet18_metrics              |   0.788154 |
| ViT-B16         | Fine-Tuning              |     0.9693 |    0.969445 |   0.9693 |   0.969322 |              11270.5  | vit_finetune_metrics          | nan        |
| ViT-B16         | Feature Extraction       |     0.9557 |    0.955738 |   0.9557 |   0.955679 |              10165.5  | vit_freeze_metrics            | nan        |

