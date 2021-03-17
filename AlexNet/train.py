
import argparse
import utils
import torch
import numpy as np
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm

from model import AlexNet
from dataloader import getLoaders

def train(args):
    device = torch.device(f"cuda:{args.device_id}")
    model = AlexNet(n_cls = 100, useLRN = args.useLRN, useDropOut= args.useDropOut)
    # model = AlexNet(num_classes= 100)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    optimizer = Adam(model.parameters(), lr = args.lr)

    train_loader , valid_loader = getLoaders(split="train", batch_size = args.batch_size, num_workers = args.num_workers, aug = args.useAug)

    train_loss_arr = []
    valid_loss_arr = []
    valid_acc_arr = []
    valid_top5_arr = []
    n_iter = 0
    best_loss = float('inf')
    best_top1_acc = 0
    best_top5_acc = 0
    for ep in range(args.epoch):
        model.train()
        for _, (img, label) in tqdm(enumerate(train_loader), total = len(train_loader)):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            # loss = model.criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss_arr.append(loss.item())
            n_iter += 1
        model.eval()
        ep_valid_loss_arr = []
        ep_acc_arr = []
        ep_top5_arr =[]
        with torch.no_grad():
            for _, (img, label) in tqdm(enumerate(valid_loader),total= len(valid_loader)):
                img, label = img.to(device), label.to(device)
                pred = model(img)
                loss = criterion(pred, label)
                # loss = model.criterion(pred, label)
                acc = utils.top_k_acc(k = 1, pred =pred.detach().cpu().numpy(),label =  label.detach().cpu().numpy())
                acc5 =  utils.top_k_acc(k = 5, pred = pred.detach().cpu().numpy(), label = label.detach().cpu().numpy())
                ep_acc_arr.append(acc)
                ep_top5_arr.append(acc5)
                ep_valid_loss_arr.append(loss.item())
        valid_loss = np.mean(ep_valid_loss_arr)
        valid_acc = np.mean(ep_acc_arr)
        valid_top5 = np.mean(ep_top5_arr)
        train_loss = np.mean(train_loss_arr[-len(train_loader):])
        valid_loss_arr.append(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_top1_acc = valid_acc
            best_top5_acc = valid_top5
            model.cpu()
            torch.save(model.state_dict(), "best_model.pth")
            model.to(device)
        if (ep+1)%10 == 0:
            model.cpu()
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss_arr,
                "valid_loss": valid_loss_arr,
                "valid_acc": valid_acc_arr,
                "valid_top5": valid_top5_arr,
                "best_loss": best_loss,
                "ep" : ep,
                "n_iter": n_iter,
            }, "model_checkpoint.pth")
            model.to(device)
        print(f"[{ep}, {n_iter}] train: {train_loss:.4f}, valid: {valid_loss:.4f}, acc: {valid_acc:.4f}, top5: {valid_top5:.4f}")
    with open("exp_result.txt","a+") as f:
        f.write(f"{args}, loss: {best_loss:.4f}, top1: {best_top1_acc*100:.1f}, top5: {best_top5_acc*100:.1f}\n")

if __name__ == "__main__":
   parser = argparse.ArgumentParser()

   parser.add_argument('--device_id', type = int, default = 0)
   parser.add_argument('--num_workers', type = int, default = 8)
   parser.add_argument('--batch_size', type = int, default = 400)

   parser.add_argument('--lr', type =float, default = 1e-4)
   parser.add_argument('--epoch', type = int, default = 50)

   parser.add_argument('--useLRN', type = str, default = "False")
   parser.add_argument('--useDropOut', type = str, default = "False")
   parser.add_argument('--useAug', type = str, default = "False")


   args = parser.parse_args()
   args.useLRN = True if args.useLRN =="True" else False
   args.useDropOut = True if args.useDropOut =="True" else False
   args.useAug = True if args.useAug =="True" else False
   train(args)