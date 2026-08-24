import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

from networks.base_model import BaseModel
from utils.cpd import (
    cpd_is_enabled,
    cpd_schedule_scale,
)
from utils.pld_lora import initialize_patchwise_discriminant_lora
from utils.training_objectives import (
    AdaptiveHardLossController,
    counterfactual_prompt_components,
    cpd_content_rejection_loss,
    cpd_diagnostics,
    cpd_direction_loss,
    fake_reweighting_loss,
    hard_real_reweighting_loss,
    symmetric_logit_anchor_diagnostics,
    symmetric_logit_anchor_loss,
)


def local_contrastive_loss(logits):
    """Compute symmetric CLIP loss before DataParallel gathers replicas."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError('contrastive logits must be a square matrix')
    targets = torch.arange(logits.shape[0], device=logits.device)
    caption_loss = F.cross_entropy(logits, targets)
    image_loss = F.cross_entropy(logits.t(), targets)
    return 0.5 * (caption_loss + image_loss)


class CLIPModel_lora(nn.Module):
    """C2P-CLIP image encoder with LoRA and one binary decision head."""

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

    def _encode_paired_prompts(self, input_ids, attention_mask):
        if input_ids.ndim != 3 or input_ids.shape[1] != 2:
            raise ValueError(
                'paired input_ids must have shape [batch, 2, sequence]')
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                'paired attention_mask must match paired input_ids')

        batch_size, prompt_count, sequence_length = input_ids.shape
        text_embeddings = F.normalize(
            self.encode_text(
                input_ids.reshape(batch_size * prompt_count, sequence_length),
                attention_mask.reshape(
                    batch_size * prompt_count, sequence_length),
            ),
            p=2,
            dim=-1,
        ).reshape(batch_size, prompt_count, -1)
        return text_embeddings

    def _encode_counterfactual_prompts(self, input_ids, attention_mask):
        text_embeddings = self._encode_paired_prompts(
            input_ids, attention_mask)
        real_embeddings, fake_embeddings = text_embeddings.unbind(dim=1)
        direction, center = counterfactual_prompt_components(
            real_embeddings, fake_embeddings)
        prompt_gap = (fake_embeddings - real_embeddings).norm(p=2, dim=-1)
        return text_embeddings, direction, center, prompt_gap

    def forward(
        self,
        images,
        input_ids=None,
        attention_mask=None,
        cla=False,
        paired_input_ids=None,
        paired_attention_mask=None,
        labels=None,
        return_cpd=False,
        return_paired_authenticity=False,
    ):
        vision_outputs = self._encode_image_outputs(images)
        adapted_features = self.model.visual_projection(
            vision_outputs.pooler_output)
        image_embeddings = F.normalize(
            adapted_features,
            p=2,
            dim=-1,
        )
        class_logits = self.model.fc(image_embeddings)
        if cla:
            return class_logits

        auxiliary = {}
        if return_paired_authenticity:
            if return_cpd:
                raise ValueError('PAPC and CPD cannot be enabled together')
            if paired_input_ids is None or paired_attention_mask is None:
                raise ValueError('paired real/fake prompts are required for PAPC')
            if labels is None:
                raise ValueError('labels are required for PAPC')
            paired_embeddings = self._encode_paired_prompts(
                paired_input_ids,
                paired_attention_mask,
            )
            label_indices = labels.flatten().long()
            if label_indices.numel() != image_embeddings.shape[0]:
                raise ValueError('PAPC labels must match the image batch')
            real_embeddings, fake_embeddings = paired_embeddings.unbind(dim=1)
            authenticity_direction = fake_embeddings - real_embeddings
            direction_norm = authenticity_direction.norm(p=2, dim=-1)
            authenticity_logits = torch.einsum(
                'bd,bpd->bp', image_embeddings, paired_embeddings)
            authenticity_logits = (
                authenticity_logits * self.model.logit_scale.exp())
            authenticity_margin = (
                authenticity_logits[:, 1] - authenticity_logits[:, 0])
            paired_authenticity_loss = F.cross_entropy(
                authenticity_logits, label_indices)
            auxiliary = {
                'paired_authenticity_margin': authenticity_margin,
                'paired_authenticity_direction_norm': direction_norm,
            }
            return (
                paired_authenticity_loss.unsqueeze(0),
                class_logits.squeeze(1),
                auxiliary,
            )
        elif return_cpd:
            if paired_input_ids is None or paired_attention_mask is None:
                raise ValueError('counterfactual prompts are required for CPD')
            if labels is None:
                raise ValueError('labels are required for CPD')
            paired_embeddings, direction, center, prompt_gap = (
                self._encode_counterfactual_prompts(
                    paired_input_ids,
                    paired_attention_mask,
                )
            )
            label_indices = labels.flatten().long()
            if label_indices.numel() != image_embeddings.shape[0]:
                raise ValueError('CPD labels must match the image batch')
            text_embeddings = paired_embeddings[
                torch.arange(
                    image_embeddings.shape[0],
                    device=image_embeddings.device,
                ),
                label_indices,
            ]
            auxiliary['cpd_direction'] = direction
            auxiliary['cpd_content_center'] = center
            auxiliary['cpd_prompt_gap'] = prompt_gap
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
        contrastive_loss = local_contrastive_loss(logits_per_text.t())
        outputs = (contrastive_loss.unsqueeze(0), class_logits.squeeze(1))
        if not return_cpd:
            return outputs

        with torch.no_grad():
            frozen_features = self.encode_image(images, disable_lora=True)
        frozen_embeddings = F.normalize(frozen_features, p=2, dim=-1)
        auxiliary['image_residual'] = (
            image_embeddings - frozen_embeddings.detach())
        return outputs + (auxiliary,)


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super().__init__(opt)
        self.delr = opt.delr
        self.claloss = opt.claloss
        self.anchor_loss_weight = opt.anchor_loss_weight
        self.logit_anchor = opt.logit_anchor
        self.cpd_direction_weight = opt.cpd_direction_weight
        self.cpd_content_weight = opt.cpd_content_weight
        self.cpd_direction_margin = opt.cpd_direction_margin
        self.cpd_start_step = opt.cpd_start_step
        self.cpd_warmup_steps = opt.cpd_warmup_steps
        self.cpd_enabled = cpd_is_enabled(opt)
        self.paired_authenticity_enabled = (
            opt.paired_authenticity_prompt_classification)
        self.hard_fake_loss_weight = opt.hard_fake_loss_weight
        self.hard_fake_fraction = opt.hard_fake_fraction
        self.fake_reweighting_mode = opt.fake_reweighting_mode
        self.hard_fake_enabled = self.hard_fake_loss_weight > 0
        self.hard_real_loss_weight = opt.hard_real_loss_weight
        self.hard_real_fraction = opt.hard_real_fraction
        self.hard_real_enabled = self.hard_real_loss_weight > 0
        self.adaptive_hard_loss_weight = opt.adaptive_hard_loss_weight
        self.adaptive_hard_temperature = opt.adaptive_hard_temperature
        self.adaptive_hard_ema_decay = opt.adaptive_hard_ema_decay
        self.adaptive_hard_warmup_steps = opt.adaptive_hard_warmup_steps
        self.adaptive_hard_enabled = self.adaptive_hard_loss_weight > 0
        self.adaptive_hard_controller = (
            AdaptiveHardLossController(
                temperature=self.adaptive_hard_temperature,
                ema_decay=self.adaptive_hard_ema_decay,
                warmup_steps=self.adaptive_hard_warmup_steps,
            )
            if self.adaptive_hard_enabled
            else None
        )
        self.pld_lora_initialization = opt.pld_lora_initialization
        self.pld_lora_initialized = False
        self.cpd_schedule_scale = 0.0
        self.effective_cpd_direction_weight = 0.0
        self.effective_cpd_content_weight = 0.0
        self.cpd_active = False
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
        if self.adaptive_hard_enabled and opt.continue_train:
            print(
                'WARNING: adaptive hard EMA state is not stored by the '
                'current model-only checkpoint format and restarts empty.')

        self.model = nn.DataParallel(
            self.model,
            device_ids=opt.gpu_ids,
        ).cuda()

    def initialize_pld_lora(self, batch):
        """Initialize LoRA from the first fixed global training batch."""
        if not self.pld_lora_initialization:
            return None
        if self.pld_lora_initialized:
            raise RuntimeError('PLD-LoRA has already been initialized')

        core_model = self.model.module
        was_training = core_model.training
        core_model.eval()
        try:
            summary = initialize_patchwise_discriminant_lora(
                core_model,
                images=batch[1],
                labels=batch[5],
                forward_images=core_model._encode_image_outputs,
                microbatch_size=self.opt.pld_lora_microbatch_size,
            )
        finally:
            core_model.train(was_training)

        self.pld_lora_initialized = True
        print(
            'PLD-LoRA initialized: '
            f'layers={summary.layers} modules={summary.modules} '
            f'rank={summary.rank} real={summary.real_samples} '
            f'fake={summary.fake_samples} '
            f'explained_energy={summary.explained_energy:.6f}'
        )
        return summary

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
        if self.cpd_enabled or self.paired_authenticity_enabled:
            if token_ids.ndim != 3 or token_ids.shape[1] != 2:
                raise ValueError(
                    'paired-prompt training requires real/fake prompt pairs')
            self.paired_input_ids = token_ids
            self.paired_attention_mask = token_attention_mask
            if self.cpd_enabled:
                batch_indices = torch.arange(
                    self.label.shape[0], device=self.label.device)
                label_indices = self.label.long()
                self.input_ids = token_ids[batch_indices, label_indices]
                self.attention_mask = token_attention_mask[
                    batch_indices, label_indices]
            else:
                self.input_ids = None
                self.attention_mask = None
        else:
            self.input_ids = token_ids
            self.attention_mask = token_attention_mask
            self.paired_input_ids = None
            self.paired_attention_mask = None

    def forward(self):
        outputs = self.model(
            self.input,
            self.input_ids,
            self.attention_mask,
            paired_input_ids=self.paired_input_ids,
            paired_attention_mask=self.paired_attention_mask,
            labels=self.label,
            return_cpd=self.cpd_active,
            return_paired_authenticity=self.paired_authenticity_enabled,
        )
        self.output, self.classhead = outputs[:2]
        auxiliary = outputs[2] if len(outputs) == 3 else {}
        self.auxiliary_components = auxiliary

    def update_cpd_schedule(self, step=None):
        """Update the delayed CPD weights for one optimizer step."""
        step = self.total_steps if step is None else step
        self.cpd_schedule_scale = cpd_schedule_scale(
            step,
            start_step=self.cpd_start_step,
            warmup_steps=self.cpd_warmup_steps,
        )
        self.effective_cpd_direction_weight = (
            self.cpd_direction_weight * self.cpd_schedule_scale)
        self.effective_cpd_content_weight = (
            self.cpd_content_weight * self.cpd_schedule_scale)
        self.cpd_active = self.cpd_enabled and (
            self.effective_cpd_direction_weight > 0
            or self.effective_cpd_content_weight > 0
        )

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.update_cpd_schedule(step=self.total_steps + 1)
        self.forward()

        self.loss_contrastive = self.output.sum()
        zero = self.classhead.new_zeros(())
        self.loss_classification = (
            self.claloss * self.loss_fn(self.classhead, self.label))
        self.loss = self.loss_contrastive + self.loss_classification

        self._update_paired_authenticity_diagnostics(zero)

        self.loss_hard_fake = zero
        if self.hard_fake_enabled:
            hard_fake_loss, hard_fake_diagnostics = (
                fake_reweighting_loss(
                    self.classhead,
                    self.label,
                    fraction=self.hard_fake_fraction,
                    mode=self.fake_reweighting_mode,
                )
            )
            self.loss_hard_fake = (
                self.claloss
                * self.hard_fake_loss_weight
                * hard_fake_loss
            )
            self.loss = self.loss + self.loss_hard_fake
            for name, value in hard_fake_diagnostics.items():
                setattr(self, name, value)

        self.loss_hard_real = zero
        if self.hard_real_enabled:
            hard_real_loss, hard_real_diagnostics = (
                hard_real_reweighting_loss(
                    self.classhead,
                    self.label,
                    fraction=self.hard_real_fraction,
                )
            )
            self.loss_hard_real = (
                self.claloss
                * self.hard_real_loss_weight
                * hard_real_loss
            )
            self.loss = self.loss + self.loss_hard_real
            for name, value in hard_real_diagnostics.items():
                setattr(self, name, value)

        self.loss_adaptive_hard = zero
        if self.adaptive_hard_enabled:
            hard_fake_budget, hard_fake_diagnostics = fake_reweighting_loss(
                self.classhead,
                self.label,
                fraction=self.hard_fake_fraction,
                mode='hard',
            )
            hard_real_budget, hard_real_diagnostics = (
                hard_real_reweighting_loss(
                    self.classhead,
                    self.label,
                    fraction=self.hard_real_fraction,
                )
            )
            for diagnostics in (
                hard_fake_diagnostics,
                hard_real_diagnostics,
            ):
                for name, value in diagnostics.items():
                    setattr(self, name, value)
            fake_share, real_share, routing_diagnostics = (
                self.adaptive_hard_controller.route(
                    self.hard_fake_bce_mean,
                    self.hard_real_bce_mean,
                    fake_selected=self.hard_fake_selected,
                    real_selected=self.hard_real_selected,
                    step=self.total_steps + 1,
                )
            )
            for name, value in routing_diagnostics.items():
                setattr(self, name, value)
            self.loss_hard_fake = (
                self.claloss
                * self.adaptive_hard_loss_weight
                * fake_share
                * hard_fake_budget
            )
            self.loss_hard_real = (
                self.claloss
                * self.adaptive_hard_loss_weight
                * real_share
                * hard_real_budget
            )
            self.loss_adaptive_hard = (
                self.loss_hard_fake + self.loss_hard_real)
            self.loss = self.loss + self.loss_adaptive_hard

        self.loss_anchor = zero
        if self.anchor_loss_weight > 0:
            self.loss_anchor = (
                self.anchor_loss_weight
                * symmetric_logit_anchor_loss(
                    self.classhead,
                    self.label,
                    anchor=self.logit_anchor,
                )
            )
            self.loss = self.loss + self.loss_anchor

        self.loss_cpd_direction = zero
        self.loss_cpd_content = zero
        if self.cpd_active:
            image_residual = self.auxiliary_components['image_residual']
            direction = self.auxiliary_components['cpd_direction']
            content_center = self.auxiliary_components['cpd_content_center']
            self.loss_cpd_direction = (
                self.effective_cpd_direction_weight
                * cpd_direction_loss(
                    image_residual,
                    direction,
                    self.label,
                    margin=self.cpd_direction_margin,
                )
            )
            self.loss_cpd_content = (
                self.effective_cpd_content_weight
                * cpd_content_rejection_loss(
                    image_residual,
                    content_center,
                )
            )
            diagnostics = cpd_diagnostics(
                image_residual,
                direction,
                content_center,
                self.label,
                self.auxiliary_components['cpd_prompt_gap'],
            )
            self.loss = (
                self.loss
                + self.loss_cpd_direction
                + self.loss_cpd_content
            )
        else:
            diagnostics = {
                'cpd_signed_projection': zero,
                'cpd_content_alignment': zero,
                'cpd_prompt_gap': zero,
            }
        for name, value in diagnostics.items():
            setattr(self, name, value)

        anchor_diagnostics = symmetric_logit_anchor_diagnostics(
            self.classhead,
            self.label,
            anchor=self.logit_anchor,
        )
        for name, value in anchor_diagnostics.items():
            setattr(self, name, value)
        self.loss.backward()
        self.optimizer.step()
        self.total_steps += 1

    def _update_paired_authenticity_diagnostics(self, zero):
        names = (
            'papc_margin_real',
            'papc_margin_fake',
            'papc_margin_std_real',
            'papc_margin_std_fake',
            'papc_direction_norm',
        )
        if not self.paired_authenticity_enabled:
            for name in names:
                setattr(self, name, zero)
            return

        margins = self.auxiliary_components[
            'paired_authenticity_margin'].detach().flatten()
        direction_norms = self.auxiliary_components[
            'paired_authenticity_direction_norm'].detach().flatten()
        real_margins = margins[self.label < 0.5]
        fake_margins = margins[self.label >= 0.5]

        def mean_and_std(values):
            if values.numel() == 0:
                return zero, zero
            return values.mean(), values.std(unbiased=False)

        self.papc_margin_real, self.papc_margin_std_real = mean_and_std(
            real_margins)
        self.papc_margin_fake, self.papc_margin_std_fake = mean_and_std(
            fake_margins)
        self.papc_direction_norm = direction_norms.mean()
