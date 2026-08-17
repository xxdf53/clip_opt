"""Train C2P-CLIP with optional training-only objectives."""

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


RETIRED_TRAINING_FLAGS = {
    '--augmentation_dro_weight',
    '--balanced_bias_calibration',
    '--classification_referenced_gradient_cap',
    '--degradation_consistency_weight',
    '--degradation_scale',
    '--ema_decay',
    '--gradient_accumulation_steps',
    '--gradient_conflict_diagnostics',
    '--gradient_conflict_projection',
    '--hard_fake_semantic_coverage',
    '--gate_loss_weight',
    '--gate_supervision_weight',
    '--gate_target_margin',
    '--init_baseline_checkpoint',
    '--freeze_global_branch',
    '--freeze_vision_lora',
    '--global_contrastive',
    '--global_contrastive_weight',
    '--boundary_center_weight',
    '--semantic_residual_weight',
    '--spectral_band_dropout',
    '--local_candidate_loss_weight',
    '--local_dim',
    '--local_dropout',
    '--local_fusion',
    '--local_gate_init',
    '--local_layer',
    '--local_pool',
    '--logit_margin',
    '--logit_center_loss_weight',
    '--margin_loss_weight',
    '--patch_residual_head',
    '--rank_loss_weight',
    '--residual_alpha',
    '--residual_scale',
    '--residual_trust_weight',
    '--residual_vib',
    '--rrsd_max_correction',
    '--symmetric_prototype_head',
    '--use_local_features',
    '--vib_beta',
    '--vib_cls_weight',
    '--vib_dim',
}


def reject_retired_training_flags(argv=None):
    """Prevent removed experiments from being silently ignored by the CLI."""
    argv = sys.argv[1:] if argv is None else argv
    used_flags = {
        argument.split('=', 1)[0]
        for argument in argv
        if argument.startswith('--')
    }
    retired_flags = sorted(used_flags & RETIRED_TRAINING_FLAGS)
    if retired_flags:
        raise ValueError(
            'retired training options are no longer supported: '
            + ', '.join(retired_flags)
        )


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
    text_loss_name = (
        'paired_authenticity'
        if getattr(model, 'paired_authenticity_enabled', False)
        else 'contrastive'
    )
    text = (
        f'loss={model.loss.item():.6f} '
        f'{text_loss_name}={model.loss_contrastive.item():.6f} '
        f'classification={model.loss_classification.item():.6f} '
        f'logit_real={model.real_logit_mean.item():.6f} '
        f'logit_fake={model.fake_logit_mean.item():.6f}'
    )
    if getattr(model, 'paired_authenticity_enabled', False):
        text += (
            f' papc_margin_real={model.papc_margin_real.item():.6f}'
            f' papc_margin_fake={model.papc_margin_fake.item():.6f}'
            f' papc_margin_std_real='
            f'{model.papc_margin_std_real.item():.6f}'
            f' papc_margin_std_fake='
            f'{model.papc_margin_std_fake.item():.6f}'
            f' papc_direction_norm={model.papc_direction_norm.item():.6f}'
        )
    if getattr(model, 'anchor_loss_weight', 0.0) > 0:
        text += (
            f' anchor={model.loss_anchor.item():.6f}'
            f' anchor_err_real={model.real_anchor_deviation.item():.6f}'
            f' anchor_err_fake={model.fake_anchor_deviation.item():.6f}'
        )
    if getattr(model, 'cpd_enabled', False):
        text += (
            f' cpd_direction={model.loss_cpd_direction.item():.6f}'
            f' cpd_content={model.loss_cpd_content.item():.6f}'
            f' cpd_scale={model.cpd_schedule_scale:.6f}'
            f' cpd_projection={model.cpd_signed_projection.item():.6f}'
            f' cpd_content_align={model.cpd_content_alignment.item():.6f}'
            f' cpd_prompt_gap={model.cpd_prompt_gap.item():.6f}'
        )
    if getattr(model, 'hard_fake_enabled', False):
        text += (
            f' hard_fake={model.loss_hard_fake.item():.6f}'
            f' hard_fake_selected={model.hard_fake_selected.item():.0f}'
            f' hard_fake_total={model.hard_fake_total.item():.0f}'
            f' hard_fake_logit_mean='
            f'{model.hard_fake_logit_mean.item():.6f}'
        )
    return text


def main():
    reject_retired_training_flags()
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
                f'total_steps * batch_size is '
                f'{expected_samples}')
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

            model.set_input(batch)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                timestamp = time.strftime(
                    '%Y_%m_%d_%H_%M_%S', time.localtime())
                print(
                    timestamp,
                    f'{format_training_losses(model)} '
                    f'optimizer_step={model.total_steps} '
                    f'lr={model.lr}',
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
