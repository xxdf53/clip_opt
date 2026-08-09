import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

from networks.base_model import BaseModel


class PatchResidualHead(nn.Module):
    """Retained only for image-only inference of existing PRH checkpoints."""

    def __init__(self, hidden_size, output_size=1, bottleneck_size=128):
        super().__init__()
        feature_size = 2 * hidden_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size, bottleneck_size),
            nn.GELU(),
            nn.Linear(bottleneck_size, output_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _patch_grid(patch_tokens):
        patch_count = patch_tokens.shape[1]
        grid_size = int(patch_count ** 0.5)
        if grid_size * grid_size != patch_count:
            raise ValueError(
                'PRH requires a square patch grid, '
                f'but received {patch_count} patch tokens'
            )
        return patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[2],
            grid_size,
            grid_size,
        )

    def forward(self, last_hidden_state):
        if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < 2:
            raise ValueError(
                'PRH expects CLIP hidden states shaped '
                '[batch, cls_plus_patches, hidden]'
            )
        patch_grid = self._patch_grid(last_hidden_state[:, 1:])
        local_mean = F.avg_pool2d(
            patch_grid,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        residual = patch_grid - local_mean
        residual_features = torch.cat(
            (
                residual.abs().mean(dim=(-2, -1)),
                residual.std(dim=(-2, -1), unbiased=False),
            ),
            dim=1,
        )
        return self.mlp(residual_features)


class SymmetricPrototypeHead(nn.Module):
    """Binary cosine classifier with an explicit zero decision boundary."""

    def __init__(self, feature_size):
        super().__init__()
        direction = F.normalize(torch.randn(feature_size), dim=0)
        self.real_prototype = nn.Parameter(-direction.clone())
        self.fake_prototype = nn.Parameter(direction.clone())

    def forward(self, image_embeddings):
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        real_prototype = F.normalize(self.real_prototype, p=2, dim=0)
        fake_prototype = F.normalize(self.fake_prototype, p=2, dim=0)
        real_similarity = image_embeddings @ real_prototype
        fake_similarity = image_embeddings @ fake_prototype
        return (fake_similarity - real_similarity).unsqueeze(1)


class CLIPModel_lora(nn.Module):
    """C2P-CLIP image encoder with LoRA and one binary decision head."""

    def __init__(
        self,
        name='openai/clip-vit-large-patch14-336',
        num_classes=1,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        patch_residual_head=False,
        symmetric_prototype_head=False,
    ):
        super().__init__()
        if patch_residual_head and symmetric_prototype_head:
            raise ValueError('PRH and SPH checkpoints cannot be combined')
        if symmetric_prototype_head and num_classes != 1:
            raise ValueError('SPH supports binary classification only')

        self.model = CLIPModel.from_pretrained(name)
        self.vision_tower = self.model.vision_model

        self.vision_tower.requires_grad_(False)
        self.model.text_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.text_projection.requires_grad_(False)
        self.model.logit_scale.requires_grad_(False)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            lora_dropout=lora_dropout,
            bias='none',
        )
        self.vision_tower_lora = get_peft_model(
            self.vision_tower, lora_config)

        projection_dim = self.model.config.projection_dim
        if symmetric_prototype_head:
            self.model.fc = SymmetricPrototypeHead(projection_dim)
        else:
            self.model.fc = nn.Linear(projection_dim, num_classes)
            nn.init.normal_(self.model.fc.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.model.fc.bias)

        self.patch_residual_head = None
        if patch_residual_head:
            self.patch_residual_head = PatchResidualHead(
                hidden_size=self.model.config.vision_config.hidden_size,
                output_size=num_classes,
            )

    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=True,
        )
        return self.model.text_projection(text_outputs.pooler_output)

    def _encode_image_outputs(self, images, disable_lora=False):
        def run_vision_tower():
            return self.vision_tower_lora(
                pixel_values=images,
                output_attentions=self.model.config.output_attentions,
                output_hidden_states=False,
                return_dict=True,
            )

        if disable_lora:
            with self.vision_tower_lora.disable_adapter():
                return run_vision_tower()
        return run_vision_tower()

    def encode_image(self, images, disable_lora=False):
        vision_outputs = self._encode_image_outputs(
            images,
            disable_lora=disable_lora,
        )
        return self.model.visual_projection(vision_outputs.pooler_output)

    def forward(
        self,
        images,
        input_ids=None,
        attention_mask=None,
        cla=False,
    ):
        vision_outputs = self._encode_image_outputs(images)
        image_embeddings = F.normalize(
            self.model.visual_projection(vision_outputs.pooler_output),
            p=2,
            dim=-1,
        )
        class_logits = self.model.fc(image_embeddings)
        patch_residual_head = getattr(self, 'patch_residual_head', None)
        if patch_residual_head is not None:
            class_logits = (
                class_logits
                + patch_residual_head(vision_outputs.last_hidden_state)
            )
        if cla:
            return class_logits

        if input_ids is None or attention_mask is None:
            raise ValueError(
                'input_ids and attention_mask are required during training')
        text_embeddings = F.normalize(
            self.encode_text(input_ids, attention_mask), p=2, dim=-1)
        logits_per_text = (
            text_embeddings @ image_embeddings.t()
            * self.model.logit_scale.exp()
        )
        return logits_per_text.t(), class_logits.squeeze(1)


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super().__init__(opt)
        self.delr = opt.delr
        self.claloss = opt.claloss
        self.model = CLIPModel_lora(
            name=opt.clip,
            lora_r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            lora_dropout=opt.lora_dropout,
            symmetric_prototype_head=opt.symmetric_prototype_head,
        )

        parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters())
        trainable_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        print(
            f'Model parameters {parameter_count:,d}; '
            f'trainable {trainable_count:,d}')

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            trainable_parameters = (
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    trainable_parameters,
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    trainable_parameters,
                    lr=opt.lr,
                    momentum=0.0,
                    weight_decay=0,
                )
            elif opt.optim == 'adamw':
                self.optimizer = torch.optim.AdamW(
                    trainable_parameters,
                    lr=opt.lr,
                    weight_decay=0.05,
                    betas=(opt.beta1, 0.999),
                    eps=1e-8,
                )
            else:
                raise ValueError('optim must be one of: adam, sgd, adamw')

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model = nn.DataParallel(self.model).cuda()

    def adjust_learning_rate(self, min_lr=1e-6):
        previous_lr = self.optimizer.param_groups[0]['lr']
        for parameter_group in self.optimizer.param_groups:
            parameter_group['lr'] *= self.delr
            if parameter_group['lr'] < min_lr:
                return False

        self.lr = self.optimizer.param_groups[0]['lr']
        print('*' * 25)
        print(
            f'Changing lr from {previous_lr} to {self.lr} '
            f'with delr {self.delr}')
        print('*' * 25)
        return True

    @staticmethod
    def _to_cuda(value):
        if isinstance(value, (tuple, list)):
            value = torch.stack(list(value))
        return value.cuda()

    def set_input(self, batch):
        self.input = batch[1].cuda()
        self.label = batch[5].cuda().float()
        self.input_ids = self._to_cuda(batch[3])
        self.attention_mask = self._to_cuda(batch[4])

    def forward(self):
        self.output, self.classhead = self.model(
            self.input,
            self.input_ids,
            self.attention_mask,
        )

    @staticmethod
    def contrastive_loss(logits):
        targets = torch.arange(len(logits), device=logits.device)
        caption_loss = F.cross_entropy(logits, targets)
        image_loss = F.cross_entropy(logits.t(), targets)
        return (caption_loss + image_loss) / 2.0

    @staticmethod
    def _masked_logit_mean(logits, mask):
        selected = logits.detach()[mask]
        if selected.numel() == 0:
            return logits.new_tensor(float('nan'))
        return selected.mean()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()

        device_logits = torch.split(
            self.output, self.output.shape[1], dim=0)
        self.loss_contrastive = sum(
            self.contrastive_loss(logits) for logits in device_logits)
        self.loss_classification = (
            self.claloss * self.loss_fn(self.classhead, self.label))
        self.loss = self.loss_contrastive + self.loss_classification
        self.real_logit_mean = self._masked_logit_mean(
            self.classhead, self.label < 0.5)
        self.fake_logit_mean = self._masked_logit_mean(
            self.classhead, self.label >= 0.5)

        self.loss.backward()
        self.optimizer.step()
        self.total_steps += 1
