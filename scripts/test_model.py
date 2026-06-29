import sys
sys.path.insert(0, '.')
import os
import time
import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from data import create_dataloader
from scripts.inference import C2P_CLIP

state_dict = torch.load('C2P_CLIP_release_20240901.pth', map_location='cpu')
model = C2P_CLIP(name='openai/clip-vit-large-patch14', num_classes=1)
model.load_state_dict(state_dict, strict=True)
model.cuda()
model.eval()


class Opt:
    dataroot = './my_first_test/'
    mode = 'binary'
    isTrain = False
    batch_size = 64
    loadSize = 224
    cropSize = 224
    no_resize = False
    no_crop = False
    no_flip = True
    serial_batches = True
    class_bal = False
    num_threads = 0
    classes = []
    rz_interp = ['bilinear']
    blur_prob = 0
    blur_sig = [0.5]
    jpg_prob = 0
    jpg_method = ['pil']
    jpg_qual = [75]
    clip = './clip-vit-large-patch14/'
    textroot = './Genimage_CNNDetection_CLIP_prefix_caption/train/'
    imgroot = './my_first_test/'


opt = Opt()
print(f'Test set: {opt.dataroot}')
data_loader = create_dataloader(opt)
print(f'Images: {len(data_loader.dataset)}')

with torch.no_grad():
    y_true, y_pred = [], []
    t0 = time.time()
    for batch in data_loader:
        img, label = batch[1], batch[5]
        y_pred.extend(model(img.cuda()).sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())
    elapsed = time.time() - t0

y_true = np.array(y_true)
y_pred = np.array(y_pred)
acc = accuracy_score(y_true, y_pred > 0.5) * 100
r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5) * 100
f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5) * 100
ap = average_precision_score(y_true.reshape(-1, 1), y_pred.reshape(-1, 1)) * 100

print(f'Elapsed: {elapsed:.1f}s')
print(f'Overall ACC: {acc:.2f}%')
print(f'Real   ACC:  {r_acc:.2f}%')
print(f'Fake   ACC:  {f_acc:.2f}%')
print(f'AP:         {ap:.2f}%')
