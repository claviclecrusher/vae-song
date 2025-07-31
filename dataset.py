import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset
import math # math 모듈 임포트 추가

# --- 가중치 생성 헬퍼 함수 ---
def _generate_weights_from_pattern(pattern, num_targets, K=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if pattern == 'uniform':
        return [1.0] * num_targets
    elif pattern == 'corner_heavy':
        # K가 제공된 경우 그리드 코너에 집중 (Grid 계열 데이터셋용)
        if K is not None and num_targets == K*K:
            weights = np.ones(num_targets, dtype=np.float32) * 0.1
            # 왼쪽 아래 코너 (0,0)에 가장 높은 가중치 부여
            weights[0] = 100.0
            # 오른쪽 아래 코너 (K-1,0)
            weights[K-1] = 50.0
            # 왼쪽 위 코너 (0, K-1)
            weights[(K-1)*K] = 50.0
            # 오른쪽 위 코너 (K-1, K-1)
            weights[K*K-1] = 20.0
            return (weights / weights.sum()).tolist()
        else: # 일반적인 컴포넌트 (SimpleGaussianMixtureDataset용)
            weights = np.ones(num_targets, dtype=np.float32) * 0.1
            # 첫 번째 컴포넌트에 높은 가중치 부여 (가장 "코너"라고 가정)
            weights[0] = 100.0
            # 마지막 컴포넌트에도 좀 더 부여
            if num_targets > 1:
                weights[num_targets - 1] = 50.0
            return (weights / weights.sum()).tolist()
    elif pattern == 'center_heavy':
        weights = np.ones(num_targets, dtype=np.float32) * 0.1
        
        if K is not None and num_targets == K*K:
            # 그리드 중앙 셀(들)에 높은 가중치 부여 (Grid 계열 데이터셋용)
            center_coords = []
            if K % 2 == 0: # 짝수 K: 중앙의 2x2 블록
                center_coords.append(((K/2)-1, (K/2)-1))
                center_coords.append(((K/2)-1, (K/2)))
                center_coords.append(((K/2), (K/2)-1))
                center_coords.append(((K/2), (K/2)))
            else: # 홀수 K: 중앙 셀
                center_coords.append((K//2, K//2))
            
            for cx, cy in center_coords:
                idx = int(cy * K + cx) # 2D 좌표를 1D 인덱스로 변환
                if 0 <= idx < num_targets:
                    weights[idx] = 100.0
        else: # 일반적인 컴포넌트 (SimpleGaussianMixtureDataset용)
            # 대략적인 중앙 컴포넌트(들)에 높은 가중치 부여
            if num_targets > 0:
                mid_idx = num_targets // 2
                weights[mid_idx] = 100.0
                if num_targets > 1 and mid_idx + 1 < num_targets:
                    weights[mid_idx + 1] = 80.0 # 옆에도 좀 더
                if num_targets > 2 and mid_idx - 1 >= 0:
                    weights[mid_idx - 1] = 80.0
        return (weights / weights.sum()).tolist()
    elif pattern == 'sparse_random':
        w = np.random.exponential(scale=1.0, size=(num_targets,))
        return (w / w.sum()).tolist()
    else:
        raise ValueError(f"알 수 없는 분포 패턴: {pattern}")


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
        total_samples (int): 전체 샘플 수
        std (float): 가우시안 표준편차
        L (float): 그리드 범위
        weights (list[float], optional): 길이 K*K, 각 셀별 샘플 비율(합 1.0). 지정 없으면 pattern에 따름.
        pattern (str, optional): 'uniform', 'corner_heavy', 'center_heavy', 'sparse_random' 중 하나
        seed (int, optional): 랜덤 시드
    """
    def __init__(self, K, total_samples, std=0.1, L=1.0, weights=None, pattern='uniform', seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.K = K
        self.L = L
        num_cells = K * K

        if weights is None:
            w = _generate_weights_from_pattern(pattern, num_cells, K=K, seed=seed)
        else:
            w = np.array(weights, dtype=np.float32)
            w = w / w.sum() # 명시된 weights도 정규화

        # 그리드 중심 좌표
        centers_x = np.linspace(0, L, K)
        centers_y = np.linspace(0, L, K)
        cell_centers = [(x, y) for x in centers_x for y in centers_y]

        # 셀별 정확한 샘플 개수 결정 (가중치 기반)
        counts = (w * total_samples).astype(int)
        remainder = total_samples - counts.sum()
        # 잔여를 랜덤하게 분배 (기존: 첫 컴포넌트에 추가 -> 변경)
        if remainder != 0:
            indices_to_add = np.random.choice(num_cells, size=abs(remainder), replace=True, p=w) # 가중치에 따라 랜덤 선택
            if remainder > 0:
                for idx in indices_to_add:
                    counts[idx] += 1
            else: # remainder < 0
                for idx in indices_to_add:
                    counts[idx] -= 1
                    if counts[idx] < 0: counts[idx] = 0 # 음수가 되지 않도록 방지
        
        # 각 셀에 대해 지정된 개수만큼 샘플 생성
        points = []
        labels = []
        for idx in range(num_cells):
            cnt = counts[idx]
            if cnt <= 0:
                continue
            cx, cy = cell_centers[idx]
            pts = np.random.randn(cnt, 2) * std + np.array([cx, cy])
            points.append(pts)
            labels.append(np.full(cnt, idx))
        
        if not points: # 생성된 포인트가 없는 경우 (예: total_samples가 너무 작을 때)
            self.X = torch.empty(0, 2, dtype=torch.float32)
            self.y = torch.empty(0, dtype=torch.int64)
        else:
            X = np.vstack(points).astype(np.float32)
            y = np.concatenate(labels).astype(np.int64)
            # 텐서화하여 속성으로 저장
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Random Gaussian Mixture Dataset ---
class RandomGaussianMixtureDataset(Dataset):
    """
    전체 공간 [0, L] 내에 num_components개의 가우시안 컴포넌트를 랜덤하게 배치하고,
    각 컴포넌트에서 불균일한 수의 샘플을 추출하는 데이터셋
    (SimpleGaussianMixtureDataset으로 대체하는 것을 권장)
    """
    # 이 클래스는 SimpleGaussianMixtureDataset으로 대체되었으므로,
    # run_vis_lip_kl_exp.py에서 이 클래스를 직접 사용하지 않도록 수정해야 합니다.
    # 이전 PR에서 SimpleGaussianMixtureDataset으로 대체된 것이 반영되지 않은 것으로 보입니다.
    # 여기서는 기존 코드를 유지하고, SimpleGaussianMixtureDataset에 패턴 로직을 추가하겠습니다.
    def __init__(self, num_components, total_samples, weights=None, std=0.1, L=1.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.num_components = num_components
        self.std = std
        self.L = L
        # 가우시안 중심을 0~L 구간에서 랜덤 샘플
        centers = np.random.uniform(0, L, size=(num_components, 2))
        # 컴포넌트별 가중치 설정
        if weights is None:
            w = np.ones(num_components, dtype=np.float32) / num_components
        else:
            w = np.array(weights, dtype=np.float32)
            w = w / w.sum()
        # 샘플 수 결정
        counts = (w * total_samples).astype(int)
        remainder = total_samples - counts.sum()
        # 잔여는 첫 컴포넌트에 추가
        if remainder > 0:
            counts[0] += remainder
        # 샘플 생성
        points, labels = [], []
        for idx in range(num_components):
            cnt = counts[idx]
            if cnt <= 0:
                continue
            mu = centers[idx]
            pts = np.random.randn(cnt, 2) * std + mu
            points.append(pts)
            labels.append(np.full(cnt, idx))
        X = np.vstack(points).astype(np.float32)
        y = np.concatenate(labels).astype(np.int64)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleGaussianMixtureDataset(Dataset):
    """
    2D 공간에 가우시안 믹스쳐로 데이터 포인트를 샘플링하는 간단한 데이터셋
    
    Args:
        num_components (int): 가우시안 컴포넌트 수
        total_samples (int): 전체 샘플 수
        centers (list[tuple], optional): 각 컴포넌트의 중심 좌표 [(x1,y1), (x2,y2), ...]. 지정 없으면 랜덤 배치
        center_range (float): centers가 None일 때 중심 좌표 생성 범위 (0~center_range)
        stds (list[float], optional): 각 컴포넌트의 표준편차 [std1, std2, ...]. 지정 없으면 기본값 적용
        weights (list[float], optional): 각 컴포넌트별 혼합 가중치 [w1, w2, ...]. 지정 없으면 pattern에 따름.
        pattern (str, optional): 'uniform', 'corner_heavy', 'center_heavy', 'sparse_random' 중 하나.
        seed (int, optional): 랜덤 시드
    """
    def __init__(self, num_components, total_samples, centers=None, center_range=4.0, stds=None, weights=None, pattern='uniform', seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.num_components = num_components
        
        # 중심 좌표 설정
        if centers is None:
            # 기본값: [0, center_range] 범위에서 랜덤하게 배치
            centers = np.random.uniform(0, center_range, size=(num_components, 2))
        else:
            centers = np.array(centers)
        
        # 표준편차 설정
        if stds is None:
            stds = [0.2] * num_components # 기본값 조정
        elif isinstance(stds, (int, float)):
            stds = [stds] * num_components
        stds = np.array(stds)
        
        # 가중치 설정
        if weights is None:
            weights = _generate_weights_from_pattern(pattern, num_components, seed=seed)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 각 컴포넌트별 샘플 수 결정
        counts = (weights * total_samples).astype(int)
        remainder = total_samples - counts.sum()
        # 잔여를 랜덤하게 분배 (기존: 첫 컴포넌트에 추가 -> 변경)
        if remainder != 0:
            indices_to_add = np.random.choice(num_components, size=abs(remainder), replace=True, p=weights) # 가중치에 따라 랜덤 선택
            if remainder > 0:
                for idx in indices_to_add:
                    counts[idx] += 1
            else: # remainder < 0
                for idx in indices_to_add:
                    counts[idx] -= 1
                    if counts[idx] < 0: counts[idx] = 0 # 음수가 되지 않도록 방지
        
        # 샘플 생성
        points = []
        labels = []
        
        for i in range(num_components):
            if counts[i] <= 0:
                continue
                
            # i번째 컴포넌트에서 샘플링
            mu = centers[i]
            std = stds[i]
            cnt = counts[i]
            
            # 가우시안 샘플링
            samples = np.random.normal(mu, std, size=(cnt, 2))
            points.append(samples)
            labels.append(np.full(cnt, i))
        
        # 모든 샘플 합치기
        if not points: # 생성된 포인트가 없는 경우
            self.X = torch.empty(0, 2, dtype=torch.float32)
            self.y = torch.empty(0, dtype=torch.int64)
        else:
            X = np.vstack(points).astype(np.float32)
            y = np.concatenate(labels).astype(np.int64)
            # 텐서로 변환
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
        
        # 파라미터 저장
        self.centers = centers
        self.stds = stds
        self.weights = weights

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




def load_dataset(dataset_name, **kwargs):
    # kwargs에서 필요한 인자들을 추출합니다.
    distribution_pattern = kwargs.get('distribution_pattern', 'uniform')
    num_components = kwargs.get('num_components', 16)
    total_samples = kwargs.get('train_total', 10000)
    std = kwargs.get('std', 0.1)
    K = kwargs.get('K', 16)
    seed = kwargs.get('seed')
    rgm_weights = kwargs.get('rgm_weights')
    rgm_total = kwargs.get('rgm_total')
    rgm_std = kwargs.get('rgm_std')
    rgm_L = kwargs.get('rgm_L')
    test_N0 = kwargs.get('test_N0')
    train_weights = kwargs.get('train_weights') # WeightedGridMixtureDataset의 명시적 weights

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
    elif dataset_name == 'grid_mixture':
        # GridMixtureDataset은 균일 분포만 가능하므로, pattern이 uniform이 아니면 WeightedGridMixtureDataset 사용
        if distribution_pattern == 'uniform' and train_weights is None:
            train_dataset = GridMixtureDataset(K, total_samples // (K*K), std=std, L=1.0) # N0 계산
        else:
            # WeightedGridMixtureDataset은 명시적 weights 또는 pattern을 통해 가중치 적용
            train_dataset = WeightedGridMixtureDataset(
                K=K, 
                total_samples=total_samples, 
                std=std, 
                L=1.0, 
                weights=train_weights, # 명시적 weights가 우선
                pattern=distribution_pattern, 
                seed=seed
            )
        # 테스트 데이터셋은 항상 균일한 GridMixtureDataset으로 생성
        test_dataset = GridMixtureDataset(K, test_N0 if test_N0 is not None else (total_samples // (K*K)), std=std, L=1.0)
    elif dataset_name == 'simple_gaussian_mixture':
        # SimpleGaussianMixtureDataset은 num_components와 rgm_ 파라미터를 사용
        train_dataset = SimpleGaussianMixtureDataset(
            num_components=num_components, 
            total_samples=rgm_total if rgm_total is not None else total_samples,
            centers=kwargs.get('rgm_centers'), # 필요하다면 centers를 직접 전달할 수 있도록
            center_range=rgm_L if rgm_L is not None else K, # rgm_L이 없으면 K를 사용
            stds=rgm_std if rgm_std is not None else std,
            weights=rgm_weights, # 명시적 rgm_weights가 우선
            pattern=distribution_pattern,
            seed=seed
        )
        # 테스트 데이터셋도 균일한 SimpleGaussianMixtureDataset으로 생성
        test_dataset = SimpleGaussianMixtureDataset(
            num_components=num_components, 
            total_samples=rgm_total if rgm_total is not None else total_samples, # 테스트도 같은 total_samples 사용
            center_range=rgm_L if rgm_L is not None else K,
            stds=rgm_std if rgm_std is not None else std,
            pattern='uniform', # 테스트는 항상 균일 분포
            seed=seed # 테스트 시드를 따로 주지 않으면 훈련과 같은 시드를 사용하여 재현성 높임
        )
    else:
        print(dataset_name, "is not implemented")
        raise NotImplementedError
    
    return train_dataset, test_dataset