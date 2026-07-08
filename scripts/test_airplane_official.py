"""
Test official C2P_CLIP_release_20240901 model on airplane (or any dataset).
Usage:
  python test_airplane_official.py
  python test_airplane_official.py --dataroot ./my_first_test/
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from data import create_dataloader
from scripts.inference import C2P_CLIP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- config ----
dataroot = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'airplane')
model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'C2P_CLIP_release_20240901.pth')

# ---- load model ----
print(f'Loading model: {model_path}')
model = C2P_CLIP(name='openai/clip-vit-large-patch14', num_classes=1)
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict, strict=True)
model.cuda()
model.eval()
print('Model loaded.')

# ---- dataloader ----
if __name__ == '__main__':
    class Opt:
        dataroot = dataroot + os.sep
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
        num_threads = 0    # 0 = 主进程加载，Windows 下避免 spawn 问题
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
    loader = create_dataloader(opt)
    print(f'Images: {len(loader.dataset)}')

    # ---- inference ----
    y_true, y_pred = [], []
    t0 = time.time()
    for batch in loader:
        img, label = batch[1], batch[5]
        y_pred.extend(model(img.cuda()).sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())
    elapsed = time.time() - t0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    real_mask = y_true == 0
    fake_mask = y_true == 1

    print(f'Elapsed:      {elapsed:.1f}s')
    print(f'Overall ACC:  {accuracy_score(y_true, y_pred > 0.5) * 100:.2f}%')
    print(f'Real   ACC:   {accuracy_score(y_true[real_mask], y_pred[real_mask] > 0.5) * 100:.2f}%')
    print(f'Fake   ACC:   {accuracy_score(y_true[fake_mask], y_pred[fake_mask] > 0.5) * 100:.2f}%')
    print(f'AP:           {average_precision_score(y_true, y_pred) * 100:.2f}%')
