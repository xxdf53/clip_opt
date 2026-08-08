import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

from networks.base_model import BaseModel
from utils.cpd import cpd_is_enabled, cpd_schedule_scale
from utils.training_objectives import (
    counterfactual_prompt_components,
    cpd_content_rejection_loss,
    cpd_diagnostics,
    cpd_direction_loss,
    symmetric_logit_anchor_diagnostics,
    symmetric_logit_anchor_loss,
    symmetric_logit_center_loss,
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

    def encode_image(self, images, disable_lora=False):
        def run_vision_tower():
            return self.vision_tower_lora(
                pixel_values=images,
                output_attentions=self.model.config.output_attentions,
                output_hidden_states=False,
                return_dict=True,
            )

        if disable_lora:
            with self.vision_tower_lora.disable_adapter():
                vision_outputs = run_vision_tower()
        else:
            vision_outputs = run_vision_tower()
        return self.model.visual_projection(vision_outputs.pooler_output)

    def _encode_counterfactual_prompts(
        self,
        input_ids,
        attention_mask,
    ):
        if input_ids.ndim != 3 or input_ids.shape[1] != 2:
            raise ValueError(
                'CPD input_ids must have shape [batch, 2, sequence]')
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                'CPD attention_mask must match counterfactual input_ids')

        batch_size, prompt_count, sequence_length = input_ids.shape
        flat_input_ids = input_ids.reshape(
            batch_size * prompt_count, sequence_length)
        flat_attention_mask = attention_mask.reshape(
            batch_size * prompt_count, sequence_length)
        text_embeddings = F.normalize(
            self.encode_text(flat_input_ids, flat_attention_mask),
            p=2,
            dim=-1,
        ).reshape(batch_size, prompt_count, -1)
        real_text_embeddings = text_embeddings[:, 0]
        fake_text_embeddings = text_embeddings[:, 1]
        authenticity_direction, content_center = (
            counterfactual_prompt_components(
                real_text_embeddings,
                fake_text_embeddings,
            )
        )
        prompt_gap = (
            fake_text_embeddings - real_text_embeddings
        ).norm(p=2, dim=-1)
        return (
            text_embeddings,
            authenticity_direction,
            content_center,
            prompt_gap,
        )

    def forward(
        self,
        images,
        input_ids=None,
        attention_mask=None,
        cla=False,
        cpd_input_ids=None,
        cpd_attention_mask=None,
        labels=None,
        return_cpd=False,
    ):
        image_embeddings = F.normalize(
            self.encode_image(images), p=2, dim=-1)
        class_logits = self.model.fc(image_embeddings)
        if cla:
            return class_logits

        cpd_components = None
        if return_cpd:
            if (
                cpd_input_ids is None
                or cpd_attention_mask is None
                or labels is None
            ):
                raise ValueError(
                    'counterfactual prompts and labels are required for CPD')
            (
                paired_text_embeddings,
                authenticity_direction,
                content_center,
                prompt_gap,
            ) = self._encode_counterfactual_prompts(
                cpd_input_ids,
                cpd_attention_mask,
            )
            label_indices = labels.flatten().long()
            if label_indices.numel() != image_embeddings.shape[0]:
                raise ValueError('CPD labels must match the image batch')
            text_embeddings = paired_text_embeddings[
                torch.arange(
                    image_embeddings.shape[0],
                    device=image_embeddings.device,
                ),
                label_indices,
            ]
            with torch.no_grad():
                base_image_embeddings = F.normalize(
                    self.encode_image(images, disable_lora=True),
                    p=2,
                    dim=-1,
                )
            cpd_components = {
                'image_residual': (
                    image_embeddings - base_image_embeddings
                ),
                'authenticity_direction': authenticity_direction,
                'content_center': content_center,
                'prompt_gap': prompt_gap,
            }
        else:
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    'input_ids and attention_mask are required during training')
            text_embeddings = F.normalize(
                self.encode_text(input_ids, attention_mask), p=2, dim=-1)

        logits_per_text = (
            text_embeddings @ image_embeddings.t()
            * self.model.logit_scale.exp()
        )
        outputs = (logits_per_text.t(), class_logits.squeeze(1))
        if return_cpd:
            return outputs + (cpd_components,)
        return outputs


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super().__init__(opt)
        self.delr = opt.delr
        self.claloss = opt.claloss
        self.anchor_loss_weight = opt.anchor_loss_weight
        self.logit_anchor = opt.logit_anchor
        self.logit_center_loss_weight = opt.logit_center_loss_weight
        self.cpd_direction_weight = opt.cpd_direction_weight
        self.cpd_content_weight = opt.cpd_content_weight
        self.cpd_direction_margin = opt.cpd_direction_margin
        self.cpd_start_step = opt.cpd_start_step
        self.cpd_warmup_steps = opt.cpd_warmup_steps
        self.cpd_enabled = cpd_is_enabled(opt)
        self.cpd_schedule_scale = 0.0
        self.effective_cpd_direction_weight = 0.0
        self.effective_cpd_content_weight = 0.0
        self.cpd_active = False

        if self.anchor_loss_weight < 0:
            raise ValueError('--anchor_loss_weight cannot be negative')
        if self.logit_anchor <= 0:
            raise ValueError('--logit_anchor must be positive')
        if self.logit_center_loss_weight < 0:
            raise ValueError('--logit_center_loss_weight cannot be negative')
        if self.cpd_direction_weight < 0:
            raise ValueError('--cpd_direction_weight cannot be negative')
        if self.cpd_content_weight < 0:
            raise ValueError('--cpd_content_weight cannot be negative')
        if self.cpd_direction_margin < 0:
            raise ValueError('--cpd_direction_margin cannot be negative')
        if self.cpd_start_step < 0:
            raise ValueError('--cpd_start_step cannot be negative')
        if self.cpd_warmup_steps < 0:
            raise ValueError('--cpd_warmup_steps cannot be negative')

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
        self.label = batch[5].cuda().float()
        token_ids = self._to_cuda(batch[3])
        token_attention_mask = self._to_cuda(batch[4])
        if self.cpd_enabled:
            if token_ids.ndim != 3 or token_ids.shape[1] != 2:
                raise ValueError(
                    'CPD training requires real/fake prompt pairs from '
                    'the training dataset')
            self.cpd_input_ids = token_ids
            self.cpd_attention_mask = token_attention_mask
            label_indices = self.label.long()
            batch_indices = torch.arange(
                self.label.shape[0], device=self.label.device)
            self.input_ids = token_ids[batch_indices, label_indices]
            self.attention_mask = token_attention_mask[
                batch_indices, label_indices]
        else:
            self.input_ids = token_ids
            self.attention_mask = token_attention_mask
            self.cpd_input_ids = None
            self.cpd_attention_mask = None

    def forward(self):
        model_outputs = self.model(
            self.input,
            self.input_ids,
            self.attention_mask,
            cpd_input_ids=self.cpd_input_ids,
            cpd_attention_mask=self.cpd_attention_mask,
            labels=self.label,
            return_cpd=self.cpd_active,
        )
        if self.cpd_active:
            self.output, self.classhead, self.cpd_components = model_outputs
        else:
            self.output, self.classhead = model_outputs
            self.cpd_components = None

    @staticmethod
    def contrastive_loss(logits):
        targets = torch.arange(len(logits), device=logits.device)
        caption_loss = F.cross_entropy(logits, targets)
        image_loss = F.cross_entropy(logits.t(), targets)
        return (caption_loss + image_loss) / 2.0

    def update_cpd_schedule(self):
        self.cpd_schedule_scale = cpd_schedule_scale(
            self.total_steps,
            start_step=self.cpd_start_step,
            warmup_steps=self.cpd_warmup_steps,
        )
        self.effective_cpd_direction_weight = (
            self.cpd_direction_weight * self.cpd_schedule_scale)
        self.effective_cpd_content_weight = (
            self.cpd_content_weight * self.cpd_schedule_scale)
        self.cpd_active = (
            self.cpd_enabled
            and (
                self.effective_cpd_direction_weight > 0
                or self.effective_cpd_content_weight > 0
            )
        )

    def optimize_parameters(self):
        self.update_cpd_schedule()
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

        if self.logit_center_loss_weight > 0:
            self.loss_logit_center = (
                self.logit_center_loss_weight
                * symmetric_logit_center_loss(
                    self.classhead,
                    self.label,
                )
            )
        else:
            self.loss_logit_center = self.classhead.new_zeros(())

        zero = self.classhead.new_zeros(())
        if self.cpd_active:
            self.loss_cpd_direction = (
                self.effective_cpd_direction_weight
                * cpd_direction_loss(
                    self.cpd_components['image_residual'],
                    self.cpd_components['authenticity_direction'],
                    self.label,
                    margin=self.cpd_direction_margin,
                )
            )
            self.loss_cpd_content = (
                self.effective_cpd_content_weight
                * cpd_content_rejection_loss(
                    self.cpd_components['image_residual'],
                    self.cpd_components['content_center'],
                )
            )
            cpd_observables = cpd_diagnostics(
                self.cpd_components['image_residual'],
                self.cpd_components['authenticity_direction'],
                self.cpd_components['content_center'],
                self.label,
                self.cpd_components['prompt_gap'],
            )
        else:
            self.loss_cpd_direction = zero
            self.loss_cpd_content = zero
            cpd_observables = {
                'cpd_signed_projection': zero,
                'cpd_content_alignment': zero,
                'cpd_prompt_gap': zero,
            }
        for name, value in cpd_observables.items():
            setattr(self, name, value)

        diagnostics = symmetric_logit_anchor_diagnostics(
            self.classhead,
            self.label,
            anchor=self.logit_anchor,
        )
        for name, value in diagnostics.items():
            setattr(self, name, value)
        self.logit_midpoint = 0.5 * (
            self.real_logit_mean + self.fake_logit_mean)

        self.loss = (
            self.loss_contrastive
            + self.loss_classification
            + self.loss_anchor
            + self.loss_logit_center
            + self.loss_cpd_direction
            + self.loss_cpd_content
        )
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
