import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import time
import logging
from tqdm import tqdm
import os

from dataloader import getDataLoader
from fcn import FCN

batch_size = 4
num_workers = 4
epoch = 100
device = torch.device("cuda:0")
lr = 1e-4
upsample = "8"
exp = 3

os.makedirs(f"Training/{exp}", exist_ok= True)

logging.basicConfig(filename=f"Training/{exp}/training.log",filemode = "w", level= logging.INFO)
logging.info(f'''batch_size = {batch_size}, epoch={epoch}, num_workers={num_workers}, lr={lr}, upsample={upsample}, GPU: RTX2070s x1''')
net = FCN(upsample=upsample, class_num=21)
net = net.to(device)

train_loader, val_loader = getDataLoader(batch_size= batch_size, num_workers = num_workers)

optimizer = Adam(net.parameters(), lr = lr,weight_decay= 5**(-4))
criterion  = nn.CrossEntropyLoss()

best_val_loss = np.inf
for ep in range(epoch):
    ep_start= time.time()
    train_loss = []
    val_loss = []
    net.train()
    for _, (img, label) in tqdm(enumerate(train_loader),total = len(train_loader)):
        optimizer.zero_grad()
        img, label = img.to(device), label.squeeze(1).to(device)
        pred = net(img)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    net.eval()
    for _, (img, label) in enumerate(val_loader):
        img, label = img.to(device), label.squeeze(1).to(device)
        pred = net(img)
        loss = criterion(pred, label)
        val_loss.append(loss.item())

    ep_end = time.time()
    if best_val_loss > np.mean(val_loss):
        best_val_loss = np.mean(val_loss)
        net.cpu()
        torch.save(net.state_dict(), f"Training/{exp}/best-FCN{upsample}.pth")
        net.to(device)
    net.cpu()
    torch.save(net.state_dict(), f"Training/{exp}/recent-FCN{upsample}.pth")
    net.to(device)
    print_string = f"[{ep+1}/{epoch}], {round(ep_end- ep_start,2)}s, train_loss: {round(np.mean(train_loss),3)}, valid_loss: {round(np.mean(val_loss),3)}"
    print(print_string)
    logging.info(print_string)


