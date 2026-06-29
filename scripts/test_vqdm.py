"""
Test C2P-CLIP on VQDM dataset
  - nature/ -> 0_real (real images)
  - ai/     -> 1_fake (AI-generated images)

Usage:
  python scripts/test_vqdm.py                    # test on val split (12K images)
  python scripts/test_vqdm.py --train            # test on train split (324K images)
  python scripts/test_vqdm.py --model C2P_CLIP_release_20240901.pth  # use other model
"""
import sys
sys.path.insert(0, '.')
import os
import subprocess
import shutil
import time
import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
from data import create_dataloader
from scripts.inference import C2P_CLIP

ROOT = r'd:\github-ware\C2P-CLIP-DeepfakeDetection'
DATASET = r'vqdm\imagenet_ai_0419_vqdm'


def parse_args():
    split = 'train' if '--train' in sys.argv else 'val'
    model_name = 'C2P_CLIP-GenImage_release_20250224.pth'
    for i, arg in enumerate(sys.argv):
        if arg == '--model' and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
    return split, model_name


def setup_test_dir(src_dir):
    """Create a clean _test dir with ONLY 0_real/1_fake junctions."""
    test_dir = src_dir + '_test'
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir)

    mapping = {'0_real': 'nature', '1_fake': 'ai'}
    for link_name, folder in mapping.items():
        target = os.path.join(src_dir, folder)
        link_path = os.path.join(test_dir, link_name)
        ret = subprocess.run(
            f'cmd /c mklink /J "{link_path}" "{target}"',
            shell=True, capture_output=True, text=True
        )
        if ret.returncode != 0:
            raise RuntimeError(f'Failed to create junction: {ret.stderr}')
        print(f'  {link_name} -> {target}')
    return test_dir


if __name__ == '__main__':
    split, model_name = parse_args()

    print(f'Dataset: {DATASET}/{split}')
    print(f'Model:   {model_name}')

    src_dir = os.path.join(ROOT, DATASET, split)
    print('Setting up test directory...')
    test_dir = setup_test_dir(src_dir)

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

    class Opt:
        dataroot = test_dir + os.sep
        mode = 'binary'
        isTrain = False
        batch_size = 128
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
        imgroot = test_dir + os.sep

    opt = Opt()
    data_loader = create_dataloader(opt)
    print(f'Images: {len(data_loader.dataset)}')

    print('Running inference...')
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
    cm = confusion_matrix(y_true, y_pred > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5) * 100
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5) * 100
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5) * 100
    ap = average_precision_score(y_true, y_pred) * 100

    print(f'\n=== Results: {DATASET}/{split} ({model_name}) ===')
    print(f'Images:      {len(y_true)}')
    print(f'Elapsed:     {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Confusion Matrix (row=true, col=pred):')
    print(f'  real pred: {cm[0,0]}, fake pred: {cm[0,1]}')
    print(f'  real pred: {cm[1,0]}, fake pred: {cm[1,1]}')
    print(f'Overall ACC: {acc:.2f}%')
    print(f'Real  ACC:   {r_acc:.2f}%')
    print(f'Fake  ACC:   {f_acc:.2f}%')
    print(f'AP:          {ap:.2f}%')
