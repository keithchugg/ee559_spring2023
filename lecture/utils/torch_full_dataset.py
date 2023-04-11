import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class FullDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset        
        self.data, self.labels = self._load_data()
        
    def _load_data(self):
        data = []
        labels = []
        
        for i in range(len(self.dataset)):
            x, y = self.dataset[i]
            data.append(x)
            labels.append(y)
        
        return torch.stack(data), torch.tensor(labels)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        