"""
Plot logit distributions of forgery features (as in the paper).
Extracts the raw logit from model's classification head for real vs fake images.

Usage:
  python scripts/plot_logit_dist.py --dataroot ./diffusion/val_test/ --model C2P_CLIP_release_20240901.pth
"""
import sys
sys.path.insert(0, '.')
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from data import create_dataloader
from scripts.inference import C2P_CLIP

ROOT = r'd:\github-ware\C2P-CLIP-DeepfakeDetection'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to test data (must have 0_real/ and 1_fake/)')
    parser.add_argument('--model', type=str, default='C2P_CLIP_release_20240901.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save', type=str, default='logit_distribution.png')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f'Loading model: {args.model}')
    model = C2P_CLIP(name=os.path.join(ROOT, 'clip-vit-large-patch14'), num_classes=1)
    state_dict = torch.load(os.path.join(ROOT, args.model), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()

    class Opt:
        dataroot = args.dataroot + os.sep
        mode = 'binary'
        isTrain = False
        batch_size = args.batch_size
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
        imgroot = args.dataroot + os.sep

    opt = Opt()
    data_loader = create_dataloader(opt)
    print(f'Images: {len(data_loader.dataset)}')

    # Collect raw logits (before sigmoid) for real and fake
    logits_real = []
    logits_fake = []

    with torch.no_grad():
        for batch in data_loader:
            img, label = batch[1], batch[5]
            logit = model(img.cuda()).flatten().cpu().tolist()
            label = label.flatten().tolist()
            for l, y in zip(logit, label):
                if y == 0:
                    logits_real.append(l)
                else:
                    logits_fake.append(l)

    logits_real = np.array(logits_real)
    logits_fake = np.array(logits_fake)

    # Plot
    plt.figure(figsize=(10, 6))
    bins = 100
    plt.hist(logits_real, bins=bins, alpha=0.6, label=f'Real (n={len(logits_real)})', color='green', edgecolor='none')
    plt.hist(logits_fake, bins=bins, alpha=0.6, label=f'Fake (n={len(logits_fake)})', color='red', edgecolor='none')
    plt.axvline(x=0.0, color='black', linestyle='--', alpha=0.5, label='Decision boundary')
    plt.xlabel('Logit (forgery feature)', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.title('Logit Distributions of Extracted Forgery Features', fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f'Saved to {args.save}')

    # Stats
    print(f'\nReal logits: mean={logits_real.mean():.3f}, std={logits_real.std():.3f}')
    print(f'Fake logits: mean={logits_fake.mean():.3f}, std={logits_fake.std():.3f}')
    sep = abs(logits_real.mean() - logits_fake.mean()) / ((logits_real.std() + logits_fake.std()) / 2)
    print(f'Separation (d\'-like): {sep:.2f}')
