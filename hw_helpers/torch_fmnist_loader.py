from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class FashionMNISTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
      image, label = self.data[index]
      return image, label

    def __len__(self):
        return len(self.data)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]) 
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

batch_size = 32

torch_data_train = FashionMNISTDataset(trainset)
trainloader = DataLoader(torch_data_train, batch_size=batch_size, shuffle=True, num_workers=2)

torch_data_test = FashionMNISTDataset(testset)
testloader = DataLoader(torch_data_test, batch_size=batch_size, shuffle=True, num_workers=2)