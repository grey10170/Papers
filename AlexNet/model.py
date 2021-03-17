import torch
import torch.nn as nn
class AlexNet(nn.Module):
    def __init__(self, n_cls = 100, useLRN = True, useDropOut = True):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3,96, (11, 11), stride = 4, padding= 2), #(224 -11 + 4)/4 + 1 = 55
            nn.ReLU(),
            nn.MaxPool2d((3,3), 2), # (55-3)/2 + 1 = 27
            nn.LocalResponseNorm(size = 5, k= 2) if useLRN else nn.Identity(),
            nn.Conv2d(96, 256, (5,5), stride = 1, padding= 2), # (27-5 +4)/1 + 1 = 27 
            nn.ReLU(),
            nn.MaxPool2d((3,3), stride = 2), #(27-3)/2 + 1 = 13
            nn.LocalResponseNorm(size = 5, k = 2) if useLRN else nn.Identity(),
            nn.Conv2d(256, 384, (3,3), stride = 1, padding= 1), # (13-3 +2)/1 + 1 = 13
            nn.ReLU(),
            nn.Conv2d(384, 384, (3,3), stride = 1, padding= 1), # (13-3 +2)/1 + 1 =  13
            nn.ReLU(),
            nn.Conv2d(384, 256, (3,3), stride =1, padding =1),# (13-3 +2)/1 + 1 =  13
            nn.ReLU(),
            nn.MaxPool2d((3,3), stride = 2), #(13-3)/2+1 = 6
        )
        self.classifier = nn.Sequential(
            nn.Dropout() if useDropOut else nn.Identity(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(),
            nn.Dropout() if useDropOut else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_cls),
        )
        self.loss_func = nn.CrossEntropyLoss()
    def forward(self, x):
        x_ = self.feature(x)
        x_ = x_.view(-1, 6*6*256)
        x_ = self.classifier(x_)
        return x_
    def pred(self, x):
        x_ = self.forward(x)
        x_ = torch.softmax(x_)
        return x_
    def criterion(self, pred, labels):
        loss = self.loss_func(pred, labels)
        return loss