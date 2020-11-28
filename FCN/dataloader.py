from torchvision import datasets
from torchvision import transforms as T

from torch.utils.data import DataLoader
import torch
import numpy as np

transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
class PILToLong:
    def __call__(self, x):
        copy = np.array(x)
        # copy[(20 < copy) & (copy < 255)] = 21
        copy[copy == 255] = 0
        return torch.tensor(copy, dtype = torch.long, requires_grad = False)
target_transform = T.Compose([
    T.Resize((256,256)),
    PILToLong()
])

def getDataLoader(batch_size = 1, num_workers =4):
    train_data = datasets.VOCSegmentation(root = "Datas", image_set = "train", download= True, transform= transform, target_transform= target_transform )
    val_data = datasets.VOCSegmentation(root = "Datas", image_set = "val", download= True, transform= transform, target_transform= target_transform)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle= True, num_workers= num_workers)
    val_loader = DataLoader(val_data, batch_size= batch_size, shuffle = False, num_workers= num_workers)

    return train_loader, val_loader


