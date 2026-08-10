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
    if opt.global_contrastive:
        configuration_parts.append('gcon')
    configuration = '__'.join(configuration_parts)
    available_base_bytes = (
        MAX_EXPERIMENT_NAME_BYTES
        - len(configuration.encode('utf-8'))
        - len('__'.encode('utf-8'))
    )
    base_name = _truncate_utf8(base_name, max_bytes=available_base_bytes)
    return f'{base_name}__{configuration}'


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
            '--global_contrastive',
            action='store_true',
            help=(
                'gather multi-GPU image/text embeddings before computing one '
                'full-batch symmetric contrastive loss'
            ),
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
        if opt.anchor_loss_weight < 0:
            raise ValueError('--anchor_loss_weight cannot be negative')
        if opt.logit_anchor <= 0:
            raise ValueError('--logit_anchor must be positive')
        if opt.cpd_direction_weight < 0:
            raise ValueError('--cpd_direction_weight cannot be negative')
        if opt.cpd_content_weight < 0:
            raise ValueError('--cpd_content_weight cannot be negative')
        if opt.cpd_direction_margin < 0:
            raise ValueError('--cpd_direction_margin cannot be negative')
        if opt.cpd_start_step < 0:
            raise ValueError('--cpd_start_step cannot be negative')
        if opt.cpd_warmup_steps < 0:
            raise ValueError('--cpd_warmup_steps cannot be negative')
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
