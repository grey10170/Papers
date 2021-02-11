from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
import torchvision.transforms as T

import os 
from PIL import Image

crop = T.Compose([
    T.RandomCrop((32,32)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.225,0.225, 0.225])
])
resize = T.Compose([
    T.Resize((32,32)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5], [0.225,0.225, 0.225])
])

class TestSet(Dataset):
    def __init__(self, data_type ,transform = None):
        super().__init__()
        self.transform = transform
        root_dir = f'./Data/{data_type}/test/'
        self.image_list = [os.path.join(root_dir, elem) for elem in os.listdir(root_dir)]
    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img , 0
    def __len__(self):
        return len(self.image_list)
def getLoader(data_type = "LSUN", transform = "Crop",batch_size = 1,epsilon = 0, n_workers = 0):
    '''
    data_type: LSUN / TinyImageNet
    transform: Crop / Resize
    '''
    if transform == "Crop":
        transform_ = crop
    elif transform == "Resize":
        transform_ = resize
    in_test_data = CIFAR100(root = "./Data", train = False, transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5], [0.225,0.225, 0.225])
    ]), download= True)
    in_test_loader = DataLoader(in_test_data, batch_size= batch_size, shuffle= False, num_workers= n_workers)
    out_test_data = TestSet(data_type, transform = transform_)
    out_test_loader = DataLoader(out_test_data, batch_size= batch_size, shuffle= False, num_workers= n_workers)
    return in_test_loader, out_test_loader
