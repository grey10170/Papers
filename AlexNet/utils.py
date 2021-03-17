import numpy as np 

def top_k_acc(k: int,pred:np.ndarray, label:np.ndarray):
    sorted_ = pred.argsort(axis = -1)
    top_k = sorted_[:, -k:]
    acc = 0
    for idx in range(top_k.shape[0]):
        if label[idx] in top_k[idx]:
            acc += 1
    return acc / top_k.shape[0]

def confusion_matrix(n_cls: int, pred:np.ndarray, label:np.ndarray):
    matrix = np.zeros((n_cls, n_cls))
    pred_idx = pred.argmax(axis = -1)
    for pred_, label_ in zip(pred_idx, label):
        matrix[label_, pred_] += 1
    return matrix