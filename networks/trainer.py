import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

from networks.base_model import BaseModel
from utils.training_objectives import (
    symmetric_logit_anchor_diagnostics,
    symmetric_logit_anchor_loss,
)


class CLIPModel_lora(nn.Module):
    """C2P-CLIP image encoder with LoRA and a binary classification head."""

    def __init__(
        self,
        name='openai/clip-vit-large-patch14-336',
        num_classes=1,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super().__init__()
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
        self.model.fc = nn.Linear(projection_dim, num_classes)
        nn.init.normal_(self.model.fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.model.fc.bias)

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

    def encode_image(self, images):
        vision_outputs = self.vision_tower_lora(
            pixel_values=images,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )
        return self.model.visual_projection(vision_outputs.pooler_output)

    def forward(self, images, input_ids=None, attention_mask=None, cla=False):
        image_embeddings = F.normalize(
            self.encode_image(images), p=2, dim=-1)
        class_logits = self.model.fc(image_embeddings)
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
        self.anchor_loss_weight = opt.anchor_loss_weight
        self.logit_anchor = opt.logit_anchor

        if self.anchor_loss_weight < 0:
            raise ValueError('--anchor_loss_weight cannot be negative')
        if self.logit_anchor <= 0:
            raise ValueError('--logit_anchor must be positive')

        self.model = CLIPModel_lora(
            name=opt.clip,
            lora_r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            lora_dropout=opt.lora_dropout,
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
        self.input_ids = self._to_cuda(batch[3])
        self.attention_mask = self._to_cuda(batch[4])
        self.label = batch[5].cuda().float()

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

    def optimize_parameters(self):
        self.forward()

        device_logits = torch.split(
            self.output, self.output.shape[1], dim=0)
        self.loss_contrastive = sum(
            self.contrastive_loss(logits) for logits in device_logits)
        self.loss_classification = (
            self.claloss * self.loss_fn(self.classhead, self.label))

        if self.anchor_loss_weight > 0:
            self.loss_anchor = (
                self.anchor_loss_weight
                * symmetric_logit_anchor_loss(
                    self.classhead,
                    self.label,
                    anchor=self.logit_anchor,
                )
            )
        else:
            self.loss_anchor = self.classhead.new_zeros(())

        diagnostics = symmetric_logit_anchor_diagnostics(
            self.classhead,
            self.label,
            anchor=self.logit_anchor,
        )
        for name, value in diagnostics.items():
            setattr(self, name, value)

        self.loss = (
            self.loss_contrastive
            + self.loss_classification
            + self.loss_anchor
        )
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
