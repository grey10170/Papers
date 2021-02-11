import torch
import numpy as np

def Softmax(x, T = 1):
    #input : batch x n_cls shape tensor
    x_ = torch.exp(x/T)
    x_ = x_/torch.sum(x_, dim= -1, keepdim= True)
    return x_

def eval_OOD(in_:list , out_:list):
    # in_, out_= np.array(in_pred_arr), np.array(out_pred_arr)
    in_.sort(), out_.sort()
    n_95 = round(len(in_)* 0.05)
    threshold = in_[n_95]
    fpr = np.sum(np.array(out_)> threshold) / len(out_)
    auroc = 0
    for idx in range(len(out_)):
        delta_ = out_[idx]
        tpr = np.sum(np.array(in_) > delta_) / len(in_)
        auroc +=tpr /len(out_)
        
    return threshold, fpr, auroc