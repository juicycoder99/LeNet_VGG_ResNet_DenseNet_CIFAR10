# CNN Architecture Comparison on CIFAR-10

A fully connected network and eight convolutional architectures trained and compared on CIFAR-10
in PyTorch, all with the same recipe (SGD + momentum, cosine LR, standard augmentation) for a fair
comparison. The full implementation and analysis is in
[`cnn_architectures_cifar10.ipynb`](cnn_architectures_cifar10.ipynb); the architectures are in
[`cifar_models.py`](cifar_models.py).

## Results (RTX 3080, 20 epochs each, GPU augmentation)

| Architecture | Params (M) | Test accuracy |
|--------------|-----------:|--------------:|
| FC Net | 3.68 | 0.499 |
| LeNet-5 | 0.08 | 0.707 |
| VGG19 | 20.0 | 0.895 |
| ResNet18 | 11.2 | 0.925 |
| SENet18 | 11.3 | **0.930** |
| ResNeXt | 5.65 | 0.900 |
| DenseNet | 1.00 | 0.925 |
| GoogLeNet | 1.45 | 0.917 |
| DPN | 2.87 | 0.918 |

The fully connected network is far behind; LeNet-5 is too shallow; and the modern convolutional
designs (residual, dense, squeeze-and-excitation, grouped convolutions) all reach ~0.90–0.93, with
the residual/dense family giving the best accuracy-per-parameter under an identical budget.

## Running it

```bash
pip install torch torchvision numpy pandas matplotlib   # CUDA build of torch for GPU
jupyter notebook cnn_architectures_cifar10.ipynb
```

CIFAR-10 is downloaded automatically by `torchvision`. `results.json` holds the per-model results.

## Files

| File | Description |
|------|-------------|
| `cnn_architectures_cifar10.ipynb` | Training, comparison and discussion |
| `cifar_models.py` | The nine CIFAR-adapted architectures |
| `results.json` | Per-model test accuracy / params / time |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
