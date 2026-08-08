"""Train C2P-CLIP with optional Logit Anchor and CPD objectives."""

import os
import random
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data import create_dataloader
from networks.trainer import Trainer
from options.test_options import TestOptions
from options.train_options import TrainOptions
from utils.evaluation_schedule import (
    should_evaluate,
    should_run_final_evaluation,
)
from utils.util import Logger
from scripts.validate import validate


def seed_torch(seed=1029):
    """Seed Python, NumPy and PyTorch for reproducible multi-GPU training."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_val_opt():
    """Build the deterministic validation options used by external callers."""
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = os.path.join(
        val_opt.dataroot, val_opt.val_split, '')
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        val_opt.blur_sig = [sum(val_opt.blur_sig) / 2]
    if len(val_opt.jpg_qual) != 1:
        val_opt.jpg_qual = [
            int((val_opt.jpg_qual[0] + val_opt.jpg_qual[-1]) / 2)
        ]
    return val_opt


def discover_evaluation_sets(test_root):
    """Return deterministic held-out subset names from the training dataset."""
    test_root = Path(test_root)
    if not test_root.is_dir():
        raise FileNotFoundError(f'test dataset directory not found: {test_root}')
    subsets = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    if not subsets:
        raise ValueError(f'no evaluation subsets found under: {test_root}')
    return subsets


def format_training_losses(model):
    return (
        f'loss={model.loss.item():.6f} '
        f'contrastive={model.loss_contrastive.item():.6f} '
        f'classification={model.loss_classification.item():.6f} '
        f'anchor={model.loss_anchor.item():.6f} '
        f'logit_center={model.loss_logit_center.item():.6f} '
        f'augmentation_dro={model.loss_augmentation_dro.item():.6f} '
        f'clean_bce={model.augmentation_group_losses[0].item():.6f} '
        f'jpeg_bce={model.augmentation_group_losses[1].item():.6f} '
        f'blur_bce={model.augmentation_group_losses[2].item():.6f} '
        f'worst_group={model.worst_augmentation_group.item():.0f} '
        f'cpd_direction={model.loss_cpd_direction.item():.6f} '
        f'cpd_content={model.loss_cpd_content.item():.6f} '
        f'cpd_scale={model.cpd_schedule_scale:.6f} '
        f'cpd_direction_weight='
        f'{model.effective_cpd_direction_weight:.6f} '
        f'cpd_projection={model.cpd_signed_projection.item():.6f} '
        f'cpd_content_align={model.cpd_content_alignment.item():.6f} '
        f'cpd_prompt_gap={model.cpd_prompt_gap.item():.6f} '
        f'logit_real={model.real_logit_mean.item():.6f} '
        f'logit_fake={model.fake_logit_mean.item():.6f} '
        f'logit_midpoint={model.logit_midpoint.item():.6f} '
        f'anchor_err_real={model.real_anchor_deviation.item():.6f} '
        f'anchor_err_fake={model.fake_anchor_deviation.item():.6f}'
    )


def main():
    opt = TrainOptions().parse()
    seed_torch(opt.seed)

    dataset_root = Path(opt.dataroot)
    test_root = dataset_root / 'test'
    evaluation_sets = discover_evaluation_sets(test_root)
    opt.dataroot = str(dataset_root / opt.train_split)

    Logger(str(Path(opt.checkpoints_dir) / opt.name / 'log.log'))
    test_opt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)
    if opt.train_manifest:
        manifest_path = Path(opt.train_manifest).resolve()
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        manifest_samples = len(data_loader.dataset)
        expected_samples = opt.total_steps * opt.batch_size
        if manifest_samples != expected_samples:
            raise ValueError(
                f'training manifest contains {manifest_samples} samples, but '
                f'total_steps * batch_size is {expected_samples}')
        print(f'training_manifest={manifest_path}')
        print(f'training_manifest_sha256={manifest_sha256}')
        print(f'training_manifest_samples={manifest_samples}')
        print(f'data_seed={opt.data_seed} model_seed={opt.seed}')
    model = Trainer(opt)

    def evaluate(epoch):
        print('*' * 25)
        print(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
        accuracies = []
        average_precisions = []

        for index, subset in enumerate(evaluation_sets):
            test_opt.dataroot = str(test_root / subset)
            test_opt.loadSize = opt.cropSize
            test_opt.cropSize = opt.cropSize
            test_opt.no_resize = False
            test_opt.no_crop = False
            test_opt.classes = ''

            accuracy, average_precision, _, _, _, _ = validate(
                model.model, test_opt)
            accuracies.append(accuracy)
            average_precisions.append(average_precision)
            print(
                f'({index} {subset:10}) '
                f'acc: {accuracy * 100:.1f}; '
                f'ap: {average_precision * 100:.1f}')

        mean_accuracy = float(np.mean(accuracies)) * 100
        mean_average_precision = float(np.mean(average_precisions)) * 100
        print(
            f'({len(evaluation_sets)} {"Mean":10}) '
            f'acc: {mean_accuracy:.1f}; ap: {mean_average_precision:.1f}')
        print('*' * 25)
        print(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
        return round(mean_accuracy, 4)

    def evaluate_and_save(epoch):
        model.eval()
        test_accuracy = evaluate(epoch)
        suffix = (
            f'{epoch}_total_steps_{model.total_steps}_'
            f'testacc_{test_accuracy}')
        model.save_networks(suffix)
        print(
            f'saving the latest model {opt.name} '
            f'(epoch {epoch}, model.total_steps {model.total_steps})')
        model.train()
        return model.total_steps

    model.train()
    last_eval_step = None
    last_epoch = 0

    for epoch in range(opt.niter):
        last_epoch = epoch
        for batch in data_loader:
            if model.total_steps >= opt.total_steps:
                break

            model.total_steps += 1
            model.set_input(batch)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                timestamp = time.strftime(
                    '%Y_%m_%d_%H_%M_%S', time.localtime())
                print(
                    timestamp,
                    f'{format_training_losses(model)} '
                    f'step={model.total_steps} lr={model.lr}',
                )

            if should_evaluate(model.total_steps, opt.eval_freq):
                print(
                    f'==========total_steps {model.total_steps}==========')
                last_eval_step = evaluate_and_save(epoch)

        if model.total_steps >= opt.total_steps:
            break

        if epoch > 0 and epoch % opt.delr_freq == 0:
            timestamp = time.strftime(
                '%Y_%m_%d_%H_%M_%S', time.localtime())
            print(
                timestamp,
                f'changing lr at the end of epoch {epoch}, '
                f'iters {model.total_steps}',
            )
            model.adjust_learning_rate()

    if should_run_final_evaluation(model.total_steps, last_eval_step):
        print(f'==========final total_steps {model.total_steps}==========')
        evaluate_and_save(last_epoch)


if __name__ == '__main__':
    main()
