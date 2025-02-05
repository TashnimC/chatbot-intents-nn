import torch
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.__n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self,  index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.__n_samples