import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import tarfile
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.utils.data import DataLoader
import torchvision

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, augmentation=None):
        self.data_dir = data_dir
        self.samples = []
        if augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.RandomRotation(degrees=0.3),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        else:
            # Substract mean of 0.1307, dive by std 0.3081
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        # Get Img path from tar
        with tarfile.open(self.data_dir, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.png'):
                    self.samples.append(member)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with tarfile.open(self.data_dir, 'r') as tar:
            member = self.samples[idx]
            f = tar.extractfile(member)
            img = Image.open(io.BytesIO(f.read()))
            img = self.transform(img)

        # Labels can be obtained from filenames: {number}_{label}.png
        file_name = member.name.split('/')[-1]
        label = int(file_name.split('_')[1].split('.')[0])
        return img, label

if __name__ == '__main__':
    def test_mnist_dataset():
        batch_size = 4

        # Dataset load
        train_dataset = MNIST('./data/train.tar',augmentation=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # first batch
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        # Display image
        print("print img shape:", images.shape)
        # Print label
        print("Labels:", labels)

    test_mnist_dataset()