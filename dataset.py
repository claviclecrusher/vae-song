import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset


class PinwheelDataset(Dataset):
    def __init__(self, radial_std, tangential_std, num_classes, num_per_class, rate):
        self.features, self.labels = self.generate_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label
        
    @staticmethod
    def generate_pinwheel_data_gpt4o(radial_std, tangential_std, num_classes, num_per_class, rate):
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = []
        labels = []
        for class_number in range(num_classes):
            r = np.random.normal(loc=1, scale=radial_std, size=num_per_class)
            t = np.random.normal(loc=class_number * 2 * np.pi / num_classes, scale=tangential_std, size=num_per_class)
            x = r * np.sin(t)
            y = r * np.cos(t)
            features.append(np.column_stack([x, y]))
            labels.append(np.full(num_per_class, class_number))

        features = np.concatenate(features).astype(np.float32)
        labels = np.concatenate(labels).astype(np.float32)
        
        angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        rotation_matrix = np.array([[np.cos(rate), -np.sin(rate)], [np.sin(rate), np.cos(rate)]])
        features = np.dot(features, rotation_matrix)
        return features, labels

    @staticmethod
    def generate_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
        # 각 클래스별 각도 계산
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = []
        labels = []
        for class_number in range(num_classes):
            # 반지름 방향 노이즈
            r = np.random.normal(loc=1, scale=radial_std, size=num_per_class)
            # 각도 방향 노이즈 (각 클래스별 중심 각도를 기준으로)
            t = np.random.normal(loc=rads[class_number], scale=tangential_std, size=num_per_class)
            
            # 극좌표를 직교좌표로 변환
            x = r * np.cos(t)  # sin과 cos 순서 변경
            y = r * np.sin(t)
            
            features.append(np.column_stack([x, y]))
            labels.append(np.full(num_per_class, class_number))

        features = np.concatenate(features).astype(np.float32)
        labels = np.concatenate(labels).astype(np.float32)
        
        # 전체 데이터셋에 회전 적용
        rotation_matrix = np.array([
            [np.cos(rate), -np.sin(rate)],
            [np.sin(rate), np.cos(rate)]
        ])
        features = np.dot(features, rotation_matrix)
        return features, labels


def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(20),
                torchvision.transforms.RandomResizedCrop((28, 28), (0.9, 1), (0.9, 1.1)),
                torchvision.transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.MNIST(root="dataset/", transform=transforms, download=True)
        test_dataset = torchvision.datasets.MNIST(root="dataset/", transform=transforms, train=False)
    elif dataset_name == 'celeba':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.CenterCrop(148),
                torchvision.transforms.Resize(64),
                torchvision.transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.CelebA(root="dataset/", transform=transforms, download=True)
        test_dataset = torchvision.datasets.CelebA(root="dataset/", transform=transforms, split="test")
    elif dataset_name == 'fashionmnist':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(28),
                torchvision.transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.FashionMNIST(root="dataset/", transform=transforms, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root="dataset/", transform=transforms, train=False)
    elif dataset_name == 'cifar10':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR10(root="dataset/", transform=transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root="dataset/", transform=transforms, train=False)
    elif dataset_name == 'omniglot':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(28),
                torchvision.transforms.ToTensor(),
            ]
        )
        train_dataset = torchvision.datasets.Omniglot(root="dataset/", transform=transforms, download=True)
        test_dataset = torchvision.datasets.Omniglot(root="dataset/", transform=transforms, background=False, download=True)
    elif dataset_name == 'pinwheel':
        train_dataset = PinwheelDataset(0.3, 0.1, 5, 1000, 0.1)
        test_dataset = PinwheelDataset(0.3, 0.1, 5, 1000, 0.1)
    else:
        print(dataset_name, "is not implemented")
        raise NotImplementedError
    
    return train_dataset, test_dataset