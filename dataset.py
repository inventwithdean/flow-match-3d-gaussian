import torch
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import  MultivariateNormal

class GaussianDataset(Dataset):
    def __init__(self):
        super().__init__()
        # 3 Different gaussian modes with each having equal probability (1/3)
        dist_0 = MultivariateNormal(torch.tensor([3.0, 3.0, -3.0]), torch.eye(3))
        dist_1 = MultivariateNormal(torch.tensor([-3.0, 0.0, -1.5]), torch.eye(3))
        dist_2 = MultivariateNormal(torch.tensor([-2.0, -2.0, 2.0]), torch.eye(3))
        points_0 = dist_0.sample((50000,))
        points_1 = dist_1.sample((50000,))
        points_2 = dist_2.sample((50000,))
        self.points = torch.cat([points_0, points_1, points_2], dim=0)

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return self.points[idx]