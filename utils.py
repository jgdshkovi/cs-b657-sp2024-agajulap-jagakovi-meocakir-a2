import numpy as np
import torch

from shuffle_vit import ShuffleViT
from simple_vit import SimpleViT


def adjust_learning_rate(optimizer, current_epoch, learning_rate):
    lr_adjust = learning_rate * (0.5 ** ((current_epoch + 1) // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_adjust
    print('Updating learning rate to {}...'.format(lr_adjust))


def model_picker(model_class):
    if model_class == 'Plain-Old-CIFAR10':
        net = SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=52, depth=6, heads=8, mlp_dim=1024)
    elif model_class == 'D-shuffletruffle':
        net = ShuffleViT(model_class=model_class, image_size=32, patch_size=2, num_classes=10, dim=52, depth=6, heads=8, mlp_ratio=4)
    elif model_class == 'N-shuffletruffle':
        net = ShuffleViT(model_class=model_class, image_size=32, patch_size=2, num_classes=10, dim=52, depth=6, heads=8, mlp_ratio=4)
    else:
        raise ValueError(f'Unknown model class: {model_class}')
    return net

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
