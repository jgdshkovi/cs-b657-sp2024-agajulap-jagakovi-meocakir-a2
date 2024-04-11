## Description

The python script main.py trains a model on CIFAR10 dataset using PyTorch. It allows you to specify the number of
epochs, model_class, batch size, learning rate, and L2 regularization strength.

## Model Performance

### Loss

| Model             | Unshuffled | 16x16 Shuffle | 8x8 Shuffle |
|-------------------|------------|---------------|-------------|
| Plain-Old-CIFAR10 | 0.78       | 2.38          | 2.89        |
| D-shuffletruffle  | 1.14       | 1.14          | 1.47        |
| N-shuffletruffle  | 1.27       | 1.27          | 1.27        |

### Accuracy

| Model             | Unshuffled | 16x16 Shuffle | 8x8 Shuffle |
|-------------------|------------|---------------|-------------|
| Plain-Old-CIFAR10 | 74.70      | 36.33         | 26.41       |
| D-shuffletruffle  | 59.77      | 59.7          | 48.34       |
| N-shuffletruffle  | 54.720     | 54.63         | 54.68       |

## PCA Analysis

### Plain-Old-CIFAR10

![pca-p-shuffle](Figures/pca_Plain-Old-CIFAR10.png "pca-p-shuffletruffle.png" )

### D-shuffletruffle

![pca-d-shuffle](Figures/pca_D-shuffletruffle.png "pca-d-shuffletruffle.png" )

### N-shuffletruffle

![pca-n-shuffle](Figures/pca_N-shuffletruffle.png "pca-n-shuffletruffle.png" )

## Sample Analysis

### Plain-Old-CIFAR10

![p-shuffle](Figures/Plain-Old-CIFAR10.png "p-shuffletruffle.png" )

### D-shuffletruffle

![d-shuffle](Figures/D-shuffletruffle.png "d-shuffletruffle.png" )

### N-shuffletruffle

![n-shuffle](Figures/N-shuffletruffle.png "n-shuffletruffle.png")

## Example Command

```sh
python main.py --epochs 10 --model_class 'Plain-Old-CIFAR10' - -batch_size 128 - -learning_rate 0.01 - -l2_regularization 0.0001
```

## Options

- epochs (int): Number of epochs for training (default: 100).
- model_class (str): Model class name. Choices - 'Plain-Old-CIFAR10','D-shuffletruffle','N-shuffletruffle'. (default: '
  Plain-Old-CIFAR10')
- batch_size (int): Batch size for training (default: 128).
- learning_rate (float): Learning rate for the optimizer (default: 0.01).
- l2_regularization (float): L2 regularization strength (default: 0.0).
