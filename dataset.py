import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset



class ChessboardDataset(Dataset):
    def __init__(self, n_data, chessboard_size=4):
        self.features, self.labels = self.generate_chessboard_data(n_data, chessboard_size)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

    @staticmethod
    def generate_chessboard_data(n_data, chessboard_size=4):
        features = np.zeros((n_data, 2))
        labels = np.zeros((n_data,))
        
        # 각 데이터 포인트에 대해
        for i in range(n_data):
            while True:
                # 0-1 사이의 랜덤한 x, y 좌표 생성
                x = np.random.random()
                y = np.random.random()
                
                # 해당 좌표가 속한 체스보드 칸 계산
                grid_x = int(x * chessboard_size)
                grid_y = int(y * chessboard_size)
                
                # 검은색 칸인지 확인 (grid_x + grid_y가 홀수인 경우가 검은색 칸)
                if (grid_x + grid_y) % 2 == 1:
                    features[i] = [x, y]
                    labels[i] = grid_x + grid_y * chessboard_size  # 칸 번호 할당
                    break
        
        return features, labels


class PinwheelDataset(Dataset):
    def __init__(self, radial_std, tangential_std, num_classes, num_per_class, rate):
        #self.features, self.labels = self.generate_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate)
        self.features, self.labels = self.generate_spin_data(num_data=10000, num_classes=num_classes)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label
        
    @staticmethod
    def generate_spin_data(num_data, num_classes, spiral=0.6):
        features = []
        labels = []
        
        points_per_class = num_data // num_classes
        
        # 나선형 제어 파라미터
        max_radius = 3.0
        spiral_factor = spiral  # 나선의 회전 정도
        noise_std = 0.1     # 노이즈 강도
        
        for class_idx in range(num_classes):
            # 기본 각도 (클래스별로 균등하게 분포)
            base_angle = 2 * np.pi * class_idx / num_classes
            
            # 반지름 값 (로그 스케일로 생성하여 안쪽에 더 많은 포인트 생성)
            radii = np.exp(np.linspace(0, np.log(max_radius), points_per_class))
            
            # 나선형을 위한 각도 계산
            # radii가 커질수록 각도가 증가하여 나선형 생성
            angles = base_angle + spiral_factor * radii
            
            # 노이즈 추가 (반지름과 각도 모두에)
            radii += np.random.normal(0, noise_std * radii, points_per_class)
            angles += np.random.normal(0, noise_std, points_per_class)
            
            # 극좌표를 데카르트 좌표로 변환
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            
            features.append(np.column_stack([x, y]))
            labels.append(np.full(points_per_class, class_idx))
        
        # 리스트를 numpy 배열로 변환
        features = np.concatenate(features).astype(np.float32)
        labels = np.concatenate(labels).astype(np.float32)
        
        # 데이터 셔플
        shuffle_idx = np.random.permutation(len(features))
        features = features[shuffle_idx]
        labels = labels[shuffle_idx]
        
        return features, labels





    @staticmethod
    def generate_pinwheel_data_regacy(radial_std, tangential_std, num_classes, num_per_class, rate):
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
    elif dataset_name == 'chessboard':
        train_dataset = ChessboardDataset(10000)
        test_dataset = ChessboardDataset(10000)
    else:
        print(dataset_name, "is not implemented")
        raise NotImplementedError
    
    return train_dataset, test_dataset