import argparse
import os
import time

import torch

import utils.util as util


MAX_EXPERIMENT_NAME_BYTES = 180


def _truncate_utf8(text, max_bytes=MAX_EXPERIMENT_NAME_BYTES):
    if max_bytes <= 0:
        return 'experiment'
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    shortened = encoded[:max_bytes].decode('utf-8', errors='ignore')
    return shortened.rstrip(' ._-') or 'experiment'


def build_experiment_name(opt, timestamp=None):
    """Build a compact, Linux-safe directory name; opt.txt keeps full config."""
    timestamp = timestamp or time.strftime('%Y%m%d-%H%M%S', time.localtime())
    base_name = ''.join(
        character if character.isalnum() or character in '-_.' else '_'
        for character in opt.name.strip()
    ) or 'experiment'
    data_seed = getattr(opt, 'data_seed', None)
    seed_part = f'ms{opt.seed}' if data_seed is not None else f's{opt.seed}'
    configuration_parts = [
        timestamp,
        *([f'ds{data_seed}'] if data_seed is not None else []),
        seed_part,
        f'r{opt.lora_r}a{opt.lora_alpha}d{opt.lora_dropout}',
        f'lr{opt.lr}',
        f'c{opt.claloss}',
    ]
    if opt.anchor_loss_weight > 0:
        configuration_parts.append(
            f'anchor-w{opt.anchor_loss_weight}-t{opt.logit_anchor}')
    if opt.cpd_direction_weight > 0 or opt.cpd_content_weight > 0:
        configuration_parts.append(
            f'cpd-d{opt.cpd_direction_weight}-c{opt.cpd_content_weight}-'
            f'm{opt.cpd_direction_margin}-s{opt.cpd_start_step}-'
            f'w{opt.cpd_warmup_steps}'
        )
    if opt.hard_fake_loss_weight > 0:
        fake_mode = getattr(opt, 'fake_reweighting_mode', 'hard')
        fake_prefix = {
            'hard': 'hfr',
            'random': 'rfr',
            'uniform': 'ufr',
        }[fake_mode]
        configuration_parts.append(
            f'{fake_prefix}-w{opt.hard_fake_loss_weight}-'
            f'q{opt.hard_fake_fraction}')
    if opt.hard_real_loss_weight > 0:
        configuration_parts.append(
            f'hrr-w{opt.hard_real_loss_weight}-q{opt.hard_real_fraction}')
    if getattr(opt, 'pld_lora_initialization', False):
        configuration_parts.append('pld-lora')
    if getattr(opt, 'paired_authenticity_prompt_classification', False):
        configuration_parts.append('papc')
    configuration = '__'.join(configuration_parts)
    available_base_bytes = (
        MAX_EXPERIMENT_NAME_BYTES
        - len(configuration.encode('utf-8'))
        - len('__'.encode('utf-8'))
    )
    base_name = _truncate_utf8(base_name, max_bytes=available_base_bytes)
    return f'{base_name}__{configuration}'


def validate_experiment_configuration(opt):
    """Validate active research options before creating output files."""
    nonnegative_options = (
        'anchor_loss_weight',
        'cpd_direction_weight',
        'cpd_content_weight',
        'cpd_direction_margin',
        'cpd_start_step',
        'cpd_warmup_steps',
        'hard_fake_loss_weight',
        'hard_real_loss_weight',
    )
    for name in nonnegative_options:
        if getattr(opt, name) < 0:
            raise ValueError(f'--{name} cannot be negative')
    if opt.logit_anchor <= 0:
        raise ValueError('--logit_anchor must be positive')
    if not 0 < opt.hard_fake_fraction < 1:
        raise ValueError('--hard_fake_fraction must be in (0, 1)')
    if opt.fake_reweighting_mode != 'hard' and (
        opt.hard_fake_loss_weight <= 0
    ):
        raise ValueError(
            '--fake_reweighting_mode requires '
            '--hard_fake_loss_weight greater than 0')
    if not 0 < opt.hard_real_fraction < 1:
        raise ValueError('--hard_real_fraction must be in (0, 1)')
    if opt.pld_lora_microbatch_size <= 0:
        raise ValueError('--pld_lora_microbatch_size must be positive')

    auxiliary_objective_enabled = (
        opt.anchor_loss_weight > 0
        or opt.cpd_direction_weight > 0
        or opt.cpd_content_weight > 0
    )
    if opt.paired_authenticity_prompt_classification and (
        auxiliary_objective_enabled
    ):
        raise ValueError(
            '--paired_authenticity_prompt_classification must be tested '
            'alone without SLAR or CPD')

    hard_reweighting_enabled = (
        opt.hard_fake_loss_weight > 0 or opt.hard_real_loss_weight > 0)
    if hard_reweighting_enabled and (
        auxiliary_objective_enabled
        or opt.paired_authenticity_prompt_classification
        or opt.pld_lora_initialization
    ):
        raise ValueError(
            'hard-example reweighting must be tested alone without '
            'PLD-LoRA, PAPC, SLAR, or CPD')

    if not opt.pld_lora_initialization:
        return
    if not opt.train_manifest:
        raise ValueError('--pld_lora_initialization requires --train_manifest')
    if getattr(opt, 'continue_train', False):
        raise ValueError('--pld_lora_initialization cannot resume a checkpoint')
    if opt.paired_authenticity_prompt_classification or (
        auxiliary_objective_enabled
    ):
        raise ValueError(
            '--pld_lora_initialization must be tested alone without '
            'PAPC, SLAR, or CPD')


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='architecture for binary classification')

        # data augmentation
        parser.add_argument('--rz_interp',       default='bilinear')
        parser.add_argument('--blur_prob',       type=float, default=0)
        parser.add_argument('--blur_sig',        default='0.5')
        parser.add_argument('--jpg_prob',        type=float, default=0)
        parser.add_argument('--jpg_method',      default='cv2')
        parser.add_argument('--jpg_qual',        default='75')

        parser.add_argument('--dataroot',        default='./dataset/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--textroot',        default='./Genimage_CNNDetection_CLIP_prefix_caption/', help='path to texts')

        parser.add_argument('--classes',         default='', help='which classes to use, separated by comma. If empty, use all subfolders of dataroot')
        parser.add_argument('--class_bal',       action='store_true')
        parser.add_argument('--batch_size',      type=int, default=64, help='input batch size')
        parser.add_argument('--keep_last_batch', action='store_true',
                            help='keep an incomplete final training batch instead of dropping it')
        parser.add_argument('--loadSize',        type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize',        type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids',         type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name',            type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch',           type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads',     type=int, default=8, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches',  action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop',  type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip',         action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type',       type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain',       type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix',          type=str,  default='', help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--delr_freq',       type=int, default=20, help='frequency of change lr')
        parser.add_argument('--delr',            type=float, default=0.8, help='delr')
        parser.add_argument('--seed',            type=int, default=123, help='seed')
        parser.add_argument(
            '--data_seed',
            type=int,
            default=None,
            help=(
                'seed reserved for DataLoader workers; use with '
                '--train_manifest to separate data randomness from --seed'
            ),
        )
        parser.add_argument(
            '--train_manifest',
            type=str,
            default='',
            help=(
                'newline-delimited image paths relative to the training root; '
                'when set, training follows this exact sample order'
            ),
        )
        parser.add_argument('--clip',            type=str, default='./clip-vit-large-patch14/', help='clip path')
        parser.add_argument('--claloss',         type=float, default=0.5, help='classification loss weight')
        parser.add_argument('--cates',           nargs='+', default=['Deepfake', 'Camera'])
        parser.add_argument('--eval_freq',       type=int, default=200, help='evaluation interval in optimizer steps; 0 disables periodic evaluation')
        parser.add_argument('--lora_r',          type=int, default=16, help='LoRA rank')
        parser.add_argument('--lora_alpha',      type=int, default=32, help='LoRA scaling parameter')
        parser.add_argument('--lora_dropout',    type=float, default=0.1, help='LoRA dropout probability')
        parser.add_argument(
            '--anchor_loss_weight',
            type=float,
            default=0.0,
            help='weight of the GAN-validated symmetric logit anchor loss',
        )
        parser.add_argument(
            '--logit_anchor',
            type=float,
            default=3.0,
            help='absolute real/fake target used by symmetric logit anchoring',
        )
        parser.add_argument(
            '--cpd_direction_weight',
            type=float,
            default=0.0,
            help='weight of counterfactual prompt direction alignment',
        )
        parser.add_argument(
            '--cpd_content_weight',
            type=float,
            default=0.0,
            help='weight of content rejection on the LoRA feature residual',
        )
        parser.add_argument(
            '--cpd_direction_margin',
            type=float,
            default=0.1,
            help='minimum signed residual projection encouraged by CPD',
        )
        parser.add_argument(
            '--cpd_start_step',
            type=int,
            default=0,
            help='optimizer step through which CPD remains disabled',
        )
        parser.add_argument(
            '--cpd_warmup_steps',
            type=int,
            default=0,
            help='steps used to linearly ramp CPD to its configured weight',
        )
        parser.add_argument(
            '--hard_fake_loss_weight',
            type=float,
            default=0.0,
            help=(
                'extra BCE weight for fake samples selected or distributed '
                'by --fake_reweighting_mode; 0 disables fake reweighting'
            ),
        )
        parser.add_argument(
            '--hard_fake_fraction',
            type=float,
            default=0.25,
            help=(
                'selected fake fraction or effective uniform-weight budget '
                'from each global batch'
            ),
        )
        parser.add_argument(
            '--fake_reweighting_mode',
            choices=('hard', 'random', 'uniform'),
            default='hard',
            help=(
                'fake auxiliary-loss selection: lowest-logit hard samples, '
                'a random count-matched subset, or a count-budget-matched '
                'uniform weight over all fake samples'
            ),
        )
        parser.add_argument(
            '--hard_real_loss_weight',
            type=float,
            default=0.0,
            help=(
                'extra BCE weight for the globally highest-logit real '
                'samples; 0 disables hard-real reweighting'
            ),
        )
        parser.add_argument(
            '--hard_real_fraction',
            type=float,
            default=0.25,
            help='fraction of real samples selected from each global batch',
        )
        parser.add_argument(
            '--paired_authenticity_prompt_classification',
            action='store_true',
            help=(
                'replace instance-level caption contrastive learning with '
                'per-image real/fake paired-prompt classification'
            ),
        )
        parser.add_argument(
            '--pld_lora_initialization',
            action='store_true',
            help=(
                'initialize vision LoRA A from patchwise real/fake '
                'activation differences in the first manifest batch'
            ),
        )
        parser.add_argument(
            '--pld_lora_microbatch_size',
            type=int,
            default=8,
            help='memory-only microbatch size for PLD-LoRA calibration',
        )
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return opt
        # return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.imgroot = opt.dataroot
        validate_experiment_configuration(opt)
        opt.name = build_experiment_name(opt)

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = _truncate_utf8(opt.name + suffix)

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
