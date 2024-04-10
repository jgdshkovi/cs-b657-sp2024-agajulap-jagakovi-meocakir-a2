# Authors:
# (based on skeleton code for CSCI-B 657, Feb 2024)
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dataset_class import PatchShuffled_CIFAR10
from shuffle_vit import ShuffleViT
from simple_vit import SimpleViT

from utils import adjust_learning_rate, EarlyStopping

def eval_model(model, data_loader, criterion, device):
    # Evaluate the model on data from valloader
    correct = 0
    total = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return val_loss / len(data_loader), 100 * correct / len(data_loader.dataset)


def main(epochs=40,
         model_class='Plain-Old-CIFAR10',
         batch_size=128,
         learning_rate=1e-3,
         l2_regularization=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Load and preprocess the dataset, feel free to add other transformations that don't shuffle the patches.
    # (Note - augmentations are typically not performed on validation set)
    transform = transforms.Compose([
        transforms.ToTensor()])

    # Initialize training, validation and test dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000],
                                                     generator=torch.Generator().manual_seed(0))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Initialize Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    train_steps = len(trainloader)

    #  Resnet
    # net = torch.hub.load('pytorch/vision:v0.17.1', 'resnet50', weights=None).to(device)
    # net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

    print(train_steps)
    # Initialize the model, the loss function and optimizer
    if model_class == 'Plain-Old-CIFAR10':
        net = ShuffleViT(model_class=model_class, image_size=32, patch_size=8, num_classes=10, dim=384, depth=12, heads=6, mlp_ratio=4).to(device)
    elif model_class == 'D-shuffletruffle':
        net = ShuffleViT(model_class=model_class, image_size=32, patch_size=2, num_classes=10, dim=52, depth=6, heads=8, mlp_ratio=4).to(device) # SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=52, depth=6, heads=8, mlp_dim=1024).to(device) #
    elif model_class == 'N-shuffletruffle':
        net = ShuffleViT(model_class=model_class, image_size=32, patch_size=2, num_classes=10, dim=52, depth=6, heads=8, mlp_ratio=4).to(device)

    print(net)  # print model architecture
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    val_acc = float('inf')
    val_loss = float('inf')
    print(f'Initial learning rate: {learning_rate}...')

    if not os.path.exists('Checkpoints'):
        os.makedirs('Checkpoints')

    early_stopping = EarlyStopping(patience=2, verbose=True)

    time_now = time.time()

    # Train the model
    try:
        for epoch in range(epochs):
            iter_count = 0
            running_loss = 0.0
            correct = 0
            net.train()
            epoch_time = time.time()
            print(f'Starting Epoch {epoch + 1}...')
            for i, data in enumerate(trainloader):
                iter_count += 1
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss:.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = int(speed * ((epochs - epoch) * train_steps - i))
                    print(f'\tspeed: {speed:.4f} s/iter; left time: {left_time}s')
                    iter_count = 0
                    time_now = time.time()

            loss = running_loss / len(trainloader)
            accuracy = 100 * correct / len(trainloader.dataset)
            cost_time = time.time() - epoch_time
            print(f'epoch - {epoch + 1} loss: {loss:.3f} accuracy: {accuracy:.3f} cost time: {cost_time:.3f}')

            if (epoch + 1) % 5 == 0:
                new_val_loss, new_val_acc = eval_model(net, valloader, criterion, device)
                print(f'\tval_loss: {val_loss:.3f} --> {new_val_loss:.3f} | val_acc: {val_acc:.3f} --> {new_val_acc:.3f}')
                val_acc = new_val_acc
                val_loss = new_val_loss
                adjust_learning_rate(optimizer, int(epoch / 5), learning_rate)
                early_stopping(val_loss, net, 'Checkpoints')
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        print('Finished training')
    except KeyboardInterrupt:
        pass

    # Load the final model
    net.load_state_dict(torch.load('Checkpoints/checkpoint.pth'))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    net.eval()
    # Evaluate the model on the test set
    test_loss, test_acc = eval_model(net, testloader, criterion, device)
    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))

    # Evaluate the model on the patch shuffled test data

    patch_size = 16
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path=f'test_patch_{patch_size}.npz', transforms=transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(
        f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')

    patch_size = 8
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path=f'test_patch_{patch_size}.npz', transforms=transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(
        f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help="number of epochs the model needs to be trained for")
    parser.add_argument('--model_class',
                        type=str,
                        default='Plain-Old-CIFAR10',
                        choices=['Plain-Old-CIFAR10', 'D-shuffletruffle', 'N-shuffletruffle'],
                        help="specifies the model class that needs to be used for training, validation and testing.")
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help="batch size for training")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.004,
                        help="learning rate for training")
    parser.add_argument('--l2_regularization',
                        type=float,
                        default=0.0,
                        help="l2 regularization for training")

    args = parser.parse_args()
    main(**vars(args))
