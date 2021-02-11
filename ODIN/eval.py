import torch
import torch.nn as nn

import argparse
from tqdm import tqdm

from Ref.densenet import densenet
from dataloader import getLoader
from utils import Softmax, eval_OOD

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)

def eval(args):
    #pretrained model loading
    device =torch.device(f"cuda:{args.device_id}")
    model = WrappedModel(densenet(num_classes = 100, depth = 190, growthRate = 40, compressionRate = 2, dropRate = 0))
    model.load_state_dict(torch.load("Ref/model_best.pth.tar")["state_dict"])
    model.eval()
    model.to(device)
    #dataloader define
    in_test_loader, out_test_loader = getLoader(data_type =args.dataset, transform= args.transform,  
                                                batch_size = args.batch_size, n_workers= args.n_workers)
    #criterion for Adversarial
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    #evaluation
    out_pred_arr = []
    in_pred_arr = []
    for idx, (image, label) in tqdm(enumerate(out_test_loader), total= len(out_test_loader)):
        image = image.to(device)
        image.requires_grad = True
        pertubed = image
        if args.epsilon > 0:
            pred = Softmax(model(image), T = args.temp)
            pred_idx = pred.argmax(dim = 1)
            loss = criterion(pred, pred_idx)
            loss.backward()
            adv = args.epsilon *torch.sign(image.grad)
            pertubed =pertubed + adv
        with torch.no_grad():
            pred = model(pertubed)
            pred = Softmax(pred, T = args.temp)
            max_ = pred.max(dim = -1)[0].detach().cpu().numpy()
            out_pred_arr += list(max_)

    for idx, (image, label) in tqdm(enumerate(in_test_loader), total= len(in_test_loader)):
        image = image.to(device)
        image.requires_grad = True
        pertubed = image
        if args.epsilon > 0:
            pred = Softmax(model(image), T= args.temp)
            pred_idx = pred.argmax(dim = 1)
            loss = criterion(pred, pred_idx)
            loss.backward()
            adv = args.epsilon *torch.sign(-image.grad)
            pertubed =pertubed + adv
        with torch.no_grad():
            pred = model(pertubed)
            pred = Softmax(pred, T = args.temp)
            max_ = pred.max(dim = -1)[0].detach().cpu().numpy()
            in_pred_arr += list(max_)
    threshold, fpr, auroc = eval_OOD(in_pred_arr, out_pred_arr)
    print(f"FPR: {fpr}, AUROC: {auroc}")
    with open("exp_log.txt",'a+') as f:
        f.write(f"{vars(args)}, FPR: {fpr}, AUROC: {auroc}\n")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type= int, default = 8)
    parser.add_argument('--device_id', type = int, default = 0)
    parser.add_argument('--n_workers', type = int, default = 4)

    parser.add_argument('--dataset', type=str, default="TinyImageNet",
                        help= "TinyImageNet / LSUN")
    parser.add_argument('--transform', type=str, default="Crop",
                        help= "Crop / Resize")
    parser.add_argument('--temp', type = float, default = 1.)
    # parser.add_argument('--threshold', type  = float, default = .5)
    parser.add_argument('--epsilon',type = float, default= 2e-3
                        , help = "FGSM-type augmentation factor")
    args = parser.parse_args()

    eval(args)

