import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from data import create_dataloader


def validate(model, opt, return_logits=False):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        y_logits = [] if return_logits else None
        for path, img, text, input_ids, attention_mask, label in data_loader:
            logits = model(img.cuda(), None, None, cla=True)
            if return_logits:
                y_logits.extend(logits.flatten().tolist())
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    result = (acc, ap, r_acc, f_acc, y_true, y_pred)
    if return_logits:
        return result + (np.asarray(y_logits),)
    return result



