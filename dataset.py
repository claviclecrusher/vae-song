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
        # vectorized sampling: generate candidates and filter by chessboard mask
        factor = 2
        num_samples = int(n_data * factor)
        X = np.random.rand(num_samples, 2)
        grid = (X * chessboard_size).astype(int)
        mask = ((grid[:, 0] + grid[:, 1]) % 2 == 1)
        X_sel = X[mask]
        # 부족하면 추가 샘플링
        while X_sel.shape[0] < n_data:
            extra = np.random.rand(n_data, 2)
            grid_e = (extra * chessboard_size).astype(int)
            mask_e = ((grid_e[:, 0] + grid_e[:, 1]) % 2 == 1)
            X_sel = np.vstack([X_sel, extra[mask_e]])
        X_sel = X_sel[:n_data]
        grid_sel = (X_sel * chessboard_size).astype(int)
        labels = (grid_sel[:, 0] + grid_sel[:, 1] * chessboard_size).astype(np.float32)
        return X_sel.astype(np.float32), labels


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


class GridMixtureDataset(Dataset):
    """
    KxK 그리드 형태로 N0개씩 가우시안 샘플을 생성하는 데이터셋
    Args:
        K (int): 그리드 차원 수
        N0 (int): 셀 당 샘플 개수
        std (float): 가우시안 표준편차
        L (float): 그리드 크기 (0~L 범위)
    """
    def __init__(self, K, N0, std=0.1, L=1.0):
        self.K = K
        self.N0 = N0
        self.std = std
        self.L = L
        # 그리드 중심점 생성
        centers_x = np.linspace(0, L, K)
        centers_y = np.linspace(0, L, K)
        points = []
        labels = []
        for idx, (cx, cy) in enumerate([(x, y) for x in centers_x for y in centers_y]):
            pts = np.random.randn(N0, 2) * std + np.array([cx, cy])
            points.append(pts)
            labels.append(np.full(N0, idx))
        X = np.vstack(points).astype(np.float32)
        y = np.concatenate(labels).astype(np.int64)
        # 텐서화하여 속성으로 저장
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 훈련용 불균일 그리드 혼합 데이터셋 클래스 추가 ---
class WeightedGridMixtureDataset(Dataset):
    """
    KxK 그리드셀에 가중치(weights)만큼 샘플을 배분하여 총 total_samples개를 생성
    Args:
        K (int): 그리드 차원 수
        weights (list[float]): 길이 K*K, 각 셀별 샘플 비율(합 1.0)
        total_samples (int): 전체 샘플 수
        std (float): 가우시안 표준편차
        L (float): 그리드 범위
    """
    def __init__(self, K, weights, total_samples, std=0.1, L=1.0):
        assert len(weights) == K*K, "weights 길이는 K*K여야 합니다"
        # store grid parameters
        self.K = K
        self.L = L
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
        # 그리드 중심 좌표
        centers_x = np.linspace(0, L, K)
        centers_y = np.linspace(0, L, K)
        cell_centers = [(x, y) for x in centers_x for y in centers_y]

        # 셀별 정확한 샘플 개수 결정 (가중치 기반)
        counts = (w * total_samples).astype(int)
        remainder = total_samples - counts.sum()
        counts[0] += remainder

        # 각 셀에 대해 지정된 개수만큼 샘플 생성
        points = []
        labels = []
        for idx in range(K*K):
            cnt = counts[idx]
            if cnt <= 0:
                continue
            cx, cy = cell_centers[idx]
            pts = np.random.randn(cnt, 2) * std + np.array([cx, cy])
            points.append(pts)
            labels.append(np.full(cnt, idx))
        X = np.vstack(points).astype(np.float32)
        y = np.concatenate(labels).astype(np.int64)
        # 텐서화하여 속성으로 저장
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(20),
            torchvision.transforms.RandomResizedCrop((28, 28), (0.9, 1), (0.9, 1.1)),
            torchvision.transforms.ToTensor(),
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.MNIST(root="dataset/", transform=train_transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root="dataset/", transform=test_transform, train=False)
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