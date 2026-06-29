"""
Test C2P_CLIP-GenImage_release_20250224 on the Chameleon dataset.

Chameleon dataset structure:
  Chameleon/test/
    0_real/   -- 14,863 real images
    1_fake/   -- 11,170 AI-generated images

Usage:
  python scripts/test_chameleon.py                        # default: C2P_CLIP-GenImage_release_20250224.pth
  python scripts/test_chameleon.py --model C2P_CLIP_release_20240901.pth   # use other checkpoint
  python scripts/test_chameleon.py --batch_size 64        # smaller batch size
"""
import sys
sys.path.insert(0, '.')
import os
import time
import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
from data import create_dataloader
from scripts.inference import C2P_CLIP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    model_name = 'C2P_CLIP-GenImage_release_20250224.pth'
    batch_size = 128
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--model' and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--batch_size' and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    return model_name, batch_size


if __name__ == '__main__':
    model_name, batch_size = parse_args()
    dataroot = os.path.join(ROOT, 'Chameleon', 'test')

    print(f'Dataset:  Chameleon/test/  (0_real + 1_fake)')
    print(f'Model:    {model_name}')
    print(f'Batch:    {batch_size}')

    # ---- Load model ----
    print('Loading model...')
    model = C2P_CLIP(name=os.path.join(ROOT, 'clip-vit-large-patch14'), num_classes=1)
    state_dict = torch.load(
        os.path.join(ROOT, model_name),
        map_location='cpu',
        weights_only=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()

    # ---- DataLoader config ----
    class Opt:
        dataroot = dataroot + os.sep
        mode = 'binary'
        isTrain = False
        batch_size = batch_size
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
        clip = os.path.join(ROOT, 'clip-vit-large-patch14')
        textroot = os.path.join(ROOT, 'Genimage_CNNDetection_CLIP_prefix_caption', 'train')
        imgroot = dataroot + os.sep

    opt = Opt()
    data_loader = create_dataloader(opt)
    print(f'Images:   {len(data_loader.dataset)}')

    # ---- Inference ----
    print('Running inference...')
    with torch.no_grad():
        y_true, y_pred = [], []
        t0 = time.time()
        for batch in data_loader:
            img, label = batch[1], batch[5]
            y_pred.extend(model(img.cuda()).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
        elapsed = time.time() - t0

    # ---- Metrics ----
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5) * 100
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5) * 100
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5) * 100
    ap = average_precision_score(y_true, y_pred) * 100

    print(f'\n===== Results: Chameleon/test  |  {model_name} =====')
    print(f'Images:         {len(y_true)}  (real: {(y_true == 0).sum()}, fake: {(y_true == 1).sum()})')
    print(f'Elapsed:        {elapsed:.1f}s  ({elapsed/60:.1f} min)')
    print(f'')
    print(f'Confusion Matrix (row=true, col=pred):')
    print(f'  TN={cm[0,0]:>6}   FP={cm[0,1]:>6}')
    print(f'  FN={cm[1,0]:>6}   TP={cm[1,1]:>6}')
    print(f'')
    print(f'Overall ACC:    {acc:.2f}%')
    print(f'Real    ACC:    {r_acc:.2f}%')
    print(f'Fake    ACC:    {f_acc:.2f}%')
    print(f'AP:             {ap:.2f}%')
    print(f'============================================')
