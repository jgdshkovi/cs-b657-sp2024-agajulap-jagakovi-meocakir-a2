import sys

import numpy as np
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn

from create_data import shuffle
from dataset_class import PatchShuffled_CIFAR10
from main import eval_model
from utils import model_picker

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def evaluate_model(net, batch_size=100):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model on the test set
    test_loss, test_acc = eval_model(net, testloader, criterion, device)
    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))

    # Evaluate the model on the patch shuffled test data
    patch_size = 16
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path=f'test_patch_{patch_size}.npz',
                                                  transforms=transform_test)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(
        f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')

    patch_size = 8
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path=f'test_patch_{patch_size}.npz',
                                                  transforms=transform_test)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(
        f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')


def sample_analysis(net):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)  # Load one by one
    samples, predictions, labels = [], [], []
    with torch.no_grad():
        for i, (data, target) in enumerate(testloader):
            if i >= 25:  # process the first 25 samples
                break
            label = target.item()
            data = data.to('cpu')
            # Shuffle images
            img_16x16 = torch.tensor(
                shuffle(16, data.clone().squeeze(0).permute(1, 2, 0).detach().numpy())).permute(2, 0,
                                                                                                        1).unsqueeze(
                0).to(device)
            img_8x8 = torch.tensor(
                shuffle(8, data.clone().squeeze(0).permute(1, 2, 0).detach().numpy())).permute(2, 0,
                                                                                                       1).unsqueeze(
                0).to(device)
            original_img = data.to(device)

            versions = [original_img, img_16x16, img_8x8]
            version_preds = []
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

            for version in versions:
                output = net(norm(version.clone()))
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                _, predicted = torch.max(output.data, 1)
                version_preds.append(predicted.item())
            samples.append([original_img.cpu(), img_16x16.cpu(), img_8x8.cpu()])
            predictions.append(version_preds)
            labels.append(label)
    _visualize_samples(samples, predictions, labels, f"{model_class} Predictions on Original and Shuffled Images",
                       model_class)


def pca_analysis(net, model_class, batch_size=100):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    features = []
    labels = []
    with torch.no_grad():
        for data in testloader:
            inputs, batch_labels = data
            outputs = net(inputs.to(device))
            features.append(outputs.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        idx = labels == i
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1], label=class_name, alpha=0.5)
    plt.title(f'PCA of {model_class}')
    plt.legend()
    filename = f'Figures/pca_{model_class}.png'
    plt.savefig(filename)
    print(f'Figure saved to {filename}')


def _extract_features_and_labels(dataloader, model):
    model.eval()
    features = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(next(model.parameters()).device)
            _, pre_final = model(inputs)
            features.append(pre_final.cpu())
            all_labels.append(labels)
    features = torch.cat(features, 0)
    all_labels = torch.cat(all_labels, 0)
    return features.numpy(), all_labels.numpy()


def _visualize_samples(samples, predictions, labels, title, model_class):
    fig, axes = plt.subplots(len(samples) + 1, len(samples[0]), figsize=(10, 2 * (len(samples) + 1)))
    column_titles = ['Original', '16x16 Shuffled', '8x8 Shuffled']
    for i, column_title in enumerate(column_titles):
        axes[0, i].text(0.5, 0.05, column_title, ha='center', va='center', transform=axes[0, i].transAxes)
        axes[0, i].axis('off')

    for i, row in enumerate(samples):
        for j, img in enumerate(row):
            ax = axes[i + 1, j]
            ax.imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())
            ax.set_title(f"Pred: {classes[predictions[i][j]]}, Label: {classes[labels[i]]}", fontsize=10)
            ax.axis('off')

    # Adjust the spacing between the rows and columns
    plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.99, bottom=0.01)
    plt.suptitle(title)
    filename = f'Figures/{model_class}.png'
    plt.savefig(filename)
    print(f'Figure saved to {filename}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_class = sys.argv[1] if len(sys.argv) > 1 else 'Plain-Old-CIFAR10'

    print(f'Evaluating {model_class}...')
    net = model_picker(model_class).to(device)

    net.load_state_dict(torch.load(f'Checkpoints/{model_class}.pth'))
    net.eval()

    evaluate_model(net)
    sample_analysis(net)
    pca_analysis(net, model_class)
