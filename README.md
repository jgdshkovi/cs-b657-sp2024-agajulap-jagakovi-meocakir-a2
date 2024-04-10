## Description

The python script main.py trains a model on CIFAR10 dataset using PyTorch. It allows you to specify the number of
epochs, model_class, batch size, learning rate, and L2 regularization strength.

## Model Performance

### Loss

| Model             | Unshuffled | 16x16 Shuffle | 8x8 Shuffle |
|-------------------|------------|---------------|-------------|
| Plain-Old-CIFAR10 |            |               |             |
| D-shuffletruffle  | 1.15       | 1.16          | 1.54        |
| N-shuffletruffle  | 1.26       | 1.26          | 1.26        |

### Accuracy

| Model             | Unshuffled | 16x16 Shuffle | 8x8 Shuffle |
|-------------------|------------|---------------|-------------|
| Plain-Old-CIFAR10 |            |               |             |
| D-shuffletruffle  | 58.96      | 58.93         | 47.06       |
| N-shuffletruffle  | 55.06      | 55.02         | 55.06       |


## PCA Analysis
### D-shuffletruffle
![pca-d-shuffle](Figures/pca_D-shuffletruffle.png "pca-D-shuffletruffle.png" ) 

### N-shuffletruffle
![pca-d-shuffle](Figures/pca_N-shuffletruffle.png "pca-D-shuffletruffle.png" ) 

## Sample Analysis

### D-shuffletruffle
![d-shuffle](Figures/D-shuffletruffle.png "D-shuffletruffle.png" ) 


### N-shuffletruffle
![n-shuffle](Figures/N-shuffletruffle.png "N-shuffletruffle.png")



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
