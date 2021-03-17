
import argparse
import utils
import torch
import numpy as np

from model import AlexNet
from tqdm import tqdm
from dataloader import getLoaders

def eval(args):
    device = torch.device(f"cuda:{args.device_id}")
    model = AlexNet(n_cls = 100)
    model.to(device)
    model.load_state_dict(torch.load(args.pretrained_path))
    model.eval()

    test_loader = getLoaders(split="eval", batch_size = args.batch_size, num_workers=args.num_workers )

    pred_arr = []
    label_arr = []
    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(test_loader),total= len(test_loader)):
            img = img.to(device)
            pred = model.pred(img)
            # mean of softmax prob from 10 different aug
            pred = pred.view(-1, 10, 100)
            pred = pred.mean(dim = 1) 
            pred_arr.append(pred.detach().cpu().numpy())
            label_arr.append(label.detach().numpy())
    pred_np = np.concatenate(pred_arr)
    label_np = np.concatenate(label_arr)
    top_1 = utils.top_k_acc(k = 1, pred = pred_np, label= label_np)
    top_5 = utils.top_k_acc(k = 5, pred = pred_np, label= label_np)
    confusion = utils.confusion_matrix(100, pred_np, label_np)
    torch.save({
        "top_1": top_1,
        "top_5": top_5,
        "confusion": confusion,
    }, "result.pth")
    print(f"top_1: {top_1*100:.2f}, top_5: {top_5*100:.2f}")
if __name__ = "__main__":
   parser = argparse.ArgumentParser()

   parser.add_argument('--device_id', type = int, default = 0)
   parser.add_argument('--num_workers', type = int, default = 4)
   parser.add_argument('--batch_size', type = int, default = 16)

   args = parser.parse_args()
   eval(args)