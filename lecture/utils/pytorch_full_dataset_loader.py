class mydataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data['x'][index,:], self.data['y'][index,:]
    def __len__(self):
        return self.data['x'].shape[0]

torch_data_train = mydataset(data_train)
dataload_train = DataLoader(torch_data_train, batch_size=batch_size, shuffle=True, num_workers=2)