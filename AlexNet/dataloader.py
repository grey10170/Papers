from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import torch
from torchvision.transforms.transforms import CenterCrop, RandomHorizontalFlip

class getTestSetting:
    def __init__(self, size):
        self.five_crop = T.FiveCrop(size)
        self.postprocess = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
        ])
    def __call__(self, sample):
        output_list = []
        list_ = self.five_crop(sample)
        for elem in list_:
            elem_ = elem.transpose(Image.FLIP_LEFT_RIGHT)
            elem = self.postprocess(elem)
            elem_ = self.postprocess(elem_)
            output_list += [elem , elem_]
        return torch.stack(output_list)
transform = T.Compose([
    T.Resize(256), #Shorter Side will matched into 256
    T.CenterCrop(256), #Crop Center to make 256x256
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
])
augmented_transform = T.Compose([
    T.Resize(256), #Shorter Side will matched into 256
    T.CenterCrop(256), #Crop Center to make 256x256
    T.RandomCrop((224,224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.225, 0.225, 0.225])
])
test_transform = getTestSetting(size = (224,224))

def getLoaders(batch_size : int,split= "train", num_workers = 0, aug = True):
    data = ImageFolder('../Data/SubImageNet/', transform = augmented_transform if aug else transform)
    train_len, valid_len = len(data)- len(data)//10 , len(data)//10
    train_data, valid_data=random_split(data,[train_len, valid_len], generator=torch.Generator().manual_seed(10170))
    if split =="train":
        train_loader = DataLoader(train_data, batch_size= batch_size, shuffle = True, num_workers= num_workers)
        valid_loader = DataLoader(valid_data, batch_size= batch_size, shuffle = False, num_workers= num_workers)
        return train_loader, valid_loader
    elif split =="eval":
        test_loader = DataLoader(valid_data, batch_size= batch_size, shuffle = False, num_workers= num_workers)
        return test_loader
    raise ValueError