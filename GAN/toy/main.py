'''
'''
import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset import toy_dataset

dataname = '25Gaussian'
noise_dim = 1

# Load Dataset
dataset = toy_dataset(dataname)
dataset = torch.Tensor(dataset)
dataset = TensorDataset(dataset)




print(len(dataset))




