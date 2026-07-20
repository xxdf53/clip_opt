import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel
from utils.checkpoint_loading import (
    extract_training_state_dict,
    select_baseline_initialization_state,
)
from utils.local_objectives import (
    confidence_preservation_loss,
    gate_sparsity_loss,
    pairwise_ranking_loss,
    relative_gate_supervision_loss,
    residual_candidate_loss,
)

from transformers import CLIPModel
from peft import LoraConfig, get_peft_model



class CLIPModel_lora(nn.Module):
    def __init__(self, name='openai/clip-vit-large-patch14-336', num_classes=1,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 use_local_features=False, local_layer=12, local_dim=256,
                 local_dropout=0.1, local_pool='mean_std',
                 local_fusion='concat', local_gate_init=0.01,
                 freeze_vision_lora=False):
        super(CLIPModel_lora, self).__init__()
        self.model        = CLIPModel.from_pretrained(name)
        self.vision_tower = self.model.vision_model
        self.vision_tower.requires_grad_(False)
        self.model.text_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.text_projection.requires_grad_(False)
        self.model.logit_scale.requires_grad_(False)
        lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=['q_proj','k_proj','v_proj'],
                lora_dropout=lora_dropout,
                bias="none",
        )
        self.vision_tower_lora = get_peft_model(self.vision_tower, lora_config)
        if freeze_vision_lora:
            self.vision_tower_lora.requires_grad_(False)

        self.use_local_features = use_local_features
        self.local_layer = local_layer
        self.local_pool = local_pool
        self.local_fusion = local_fusion
        vision_dim = self.model.config.vision_config.hidden_size
        projection_dim = self.model.config.projection_dim

        if self.use_local_features:
            num_layers = self.model.config.vision_config.num_hidden_layers
            if not 1 <= self.local_layer <= num_layers:
                raise ValueError(
                    f'local_layer must be in [1, {num_layers}], got {self.local_layer}')
            if self.local_pool not in ('mean', 'mean_std'):
                raise ValueError(
                    f"local_pool must be 'mean' or 'mean_std', got {self.local_pool}")
            if self.local_fusion not in (
                    'concat', 'residual_gate', 'adaptive_residual'):
                raise ValueError(
                    'unsupported local_fusion: '
                    f'got {self.local_fusion}')
            if (self.local_fusion != 'concat'
                    and not 0.0 < local_gate_init < 1.0):
                raise ValueError(
                    f'local_gate_init must be in (0, 1), got {local_gate_init}')

            pooled_dim = vision_dim if self.local_pool == 'mean' else 2 * vision_dim
            self.local_norm = nn.LayerNorm(vision_dim)
            self.local_projector = nn.Sequential(
                nn.Linear(pooled_dim, local_dim),
                nn.GELU(),
                nn.Dropout(local_dropout),
            )
            if self.local_fusion == 'concat':
                # Legacy local-feature head. Keep this path unchanged so that
                # checkpoints trained before residual gating remain loadable.
                self.model.fc = nn.Sequential(
                    nn.Linear(projection_dim + local_dim, 256),
                    nn.GELU(),
                    nn.Dropout(local_dropout),
                    nn.Linear(256, num_classes),
                )
            else:
                self.model.fc = nn.Linear(projection_dim, num_classes)
                self.local_classifier = nn.Linear(local_dim, num_classes)
                gate_probability = torch.tensor(
                    float(local_gate_init), dtype=torch.float32)
                gate_bias = torch.logit(gate_probability)
                if self.local_fusion == 'residual_gate':
                    # Compatibility path for the first scalar-gate experiment.
                    self.local_gate_logit = nn.Parameter(gate_bias)
                else:
                    gate_input_dim = projection_dim + local_dim + 2
                    self.local_gate_network = nn.Linear(gate_input_dim, 1)
                    nn.init.zeros_(self.local_gate_network.weight)
                    nn.init.constant_(
                        self.local_gate_network.bias, gate_bias.item())
        else:
            self.model.fc = nn.Linear(projection_dim, num_classes)

        self.model.fc.apply(self._init_linear)
        if self.use_local_features:
            self.local_projector.apply(self._init_linear)
            if self.local_fusion in ('residual_gate', 'adaptive_residual'):
                self.local_classifier.apply(self._init_linear)

    @staticmethod
    def _init_linear(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions   = self.model.config.output_attentions,
            output_hidden_states= self.model.config.output_hidden_states,
            return_dict         = self.model.config.use_return_dict, 
        )
        text_embeds = text_outputs[1]
        text_embeds = self.model.text_projection(text_embeds)
        return text_embeds
    
    def encode_image(self, img):
        vision_outputs = self.vision_tower_lora(
            pixel_values=img,
            output_attentions    = self.model.config.output_attentions,
            output_hidden_states = self.use_local_features,
            return_dict          = True,
        )
        image_features = self.model.visual_projection(
            vision_outputs.pooler_output)

        if not self.use_local_features:
            return image_features, None

        # hidden_states[0] is the patch embedding output, so index k is the
        # representation after the k-th Transformer layer.
        local_tokens = vision_outputs.hidden_states[self.local_layer][:, 1:, :]
        local_tokens = self.local_norm(local_tokens)
        local_mean = local_tokens.mean(dim=1)
        if self.local_pool == 'mean_std':
            local_std = local_tokens.std(dim=1, unbiased=False)
            local_stats = torch.cat((local_mean, local_std), dim=-1)
        else:
            local_stats = local_mean
        local_features = self.local_projector(local_stats)
        return image_features, local_features

    def classification_outputs(self, image_features, local_features,
                               gate_override=None):
        image_features = F.normalize(image_features, p=2, dim=-1)
        if not self.use_local_features:
            global_logits = self.model.fc(image_features)
            return image_features, {
                'final_logits': global_logits,
                'global_logits': global_logits,
            }

        local_features = F.normalize(local_features, p=2, dim=-1)
        if self.local_fusion == 'concat':
            fused_features = torch.cat((image_features, local_features), dim=-1)
            return image_features, {
                'final_logits': self.model.fc(fused_features),
            }

        global_logits = self.model.fc(image_features)
        local_logits = self.local_classifier(local_features)
        if self.local_fusion == 'residual_gate':
            learned_gate = torch.sigmoid(self.local_gate_logit).expand_as(
                global_logits)
        else:
            gate_inputs = torch.cat((
                image_features.detach(),
                local_features.detach(),
                global_logits.detach().abs(),
                (local_logits.detach() - global_logits.detach()).abs(),
            ), dim=-1)
            learned_gate = torch.sigmoid(
                self.local_gate_network(gate_inputs))

        if gate_override is None:
            gate = learned_gate
        else:
            if not 0.0 <= float(gate_override) <= 1.0:
                raise ValueError('gate_override must be in [0, 1]')
            gate = torch.full_like(learned_gate, float(gate_override))

        final_logits = global_logits + gate * local_logits
        outputs = {
            'final_logits': final_logits,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'gate': gate,
        }
        if gate_override is not None:
            outputs['learned_gate'] = learned_gate
        return image_features, outputs

    def classify(self, image_features, local_features, gate_override=None):
        image_features, outputs = self.classification_outputs(
            image_features, local_features, gate_override=gate_override)
        return image_features, outputs['final_logits']

    def forward_components(self, img, gate_override=None):
        image_features, local_features = self.encode_image(img)
        _, outputs = self.classification_outputs(
            image_features, local_features, gate_override=gate_override)
        return outputs

    def local_gate_value(self):
        """Return the legacy scalar gate; adaptive gates require an image."""
        if not self.use_local_features or self.local_fusion != 'residual_gate':
            return None
        return torch.sigmoid(self.local_gate_logit)

    def freeze_global_parameters(self):
        """Protect the initialized baseline while training local corrections."""
        self.vision_tower_lora.requires_grad_(False)
        self.model.fc.requires_grad_(False)
    
    def forward(self, img, input_ids, attention_mask, cla=False,
                return_components=False, gate_override=None):
        # tmp = x; print(f'x: {tmp.shape}, max: {tmp.max()}, min: {tmp.min()}, mean: {tmp.mean()}')

        image_features, local_features = self.encode_image(img)
        image_embeds, outputs = self.classification_outputs(
            image_features, local_features, gate_override=gate_override)
        classhead = outputs['final_logits']
        if cla:
            return outputs if return_components else classhead
        
        text_embeds  = self.encode_text(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.model.logit_scale.exp()
        logits_per_image = logits_per_text.t()

        if return_components:
            return logits_per_image, classhead.squeeze(1), outputs
        return logits_per_image, classhead.squeeze(1)
    
    def forward_eval(self, img, gate_override=None):
        return self.forward_components(
            img, gate_override=gate_override)['final_logits']

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.delr = opt.delr
        self.claloss = opt.claloss
        self.rank_loss_weight = opt.rank_loss_weight
        self.preserve_loss_weight = opt.preserve_loss_weight
        self.gate_loss_weight = opt.gate_loss_weight
        self.local_candidate_loss_weight = opt.local_candidate_loss_weight
        self.gate_supervision_weight = opt.gate_supervision_weight
        self.gate_target_margin = opt.gate_target_margin
        self.freeze_global_branch = opt.freeze_global_branch
        
        self.model = CLIPModel_lora(
            name=opt.clip,
            lora_r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            lora_dropout=opt.lora_dropout,
            use_local_features=opt.use_local_features,
            local_layer=opt.local_layer,
            local_dim=opt.local_dim,
            local_dropout=opt.local_dropout,
            local_pool=opt.local_pool,
            local_fusion=opt.local_fusion,
            local_gate_init=opt.local_gate_init,
            freeze_vision_lora=opt.freeze_vision_lora,
        )

        if opt.init_baseline_checkpoint:
            payload = torch.load(
                opt.init_baseline_checkpoint,
                map_location='cpu',
                weights_only=True,
            )
            baseline_state, baseline_steps = extract_training_state_dict(payload)
            compatible_state = select_baseline_initialization_state(
                baseline_state, self.model.state_dict())
            self.model.load_state_dict(compatible_state, strict=False)
            print(
                f'Initialized global branch from {opt.init_baseline_checkpoint} '
                f'({len(compatible_state)} tensors, '
                f'total_steps={baseline_steps if baseline_steps is not None else "unknown"})')

        if self.freeze_global_branch:
            if not opt.init_baseline_checkpoint and not opt.continue_train:
                raise ValueError(
                    '--freeze_global_branch requires a baseline checkpoint '
                    'for a new run')
            self.model.freeze_global_parameters()

        auxiliary_weights = (
            self.preserve_loss_weight,
            self.gate_loss_weight,
            self.local_candidate_loss_weight,
            self.gate_supervision_weight,
        )
        if any(weight < 0 for weight in (
                self.rank_loss_weight,) + auxiliary_weights):
            raise ValueError('auxiliary loss weights cannot be negative')
        if (any(auxiliary_weights) and (
                not opt.use_local_features
                or opt.local_fusion != 'adaptive_residual')):
            raise ValueError(
                'local auxiliary losses require --use_local_features and '
                '--local_fusion adaptive_residual')
        if self.gate_target_margin <= 0:
            raise ValueError('--gate_target_margin must be positive')

        net_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Model parameters {net_params:,d}; trainable {trainable_params:,d}')

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            elif opt.optim == 'adamw':
                self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, weight_decay=0.05, betas=(opt.beta1, 0.999), eps=1e-8)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        
        self.model = nn.DataParallel(self.model).cuda()

 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.delr
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/self.delr} to {param_group["lr"]} with delr {self.delr}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input           = input[1].cuda()
        if self.freeze_global_branch:
            self.input_ids = None
            self.attention_mask = None
            self.label = input[5].cuda().float()
            return
        # Handle multiprocessing collate edge case: input_ids / attention_mask
        # may arrive as tuples of tensors instead of stacked tensors
        if isinstance(input[3], (tuple, list)):
            self.input_ids = torch.stack(list(input[3])).cuda()
        else:
            self.input_ids = input[3].cuda()
        if isinstance(input[4], (tuple, list)):
            self.attention_mask = torch.stack(list(input[4])).cuda()
        else:
            self.attention_mask = input[4].cuda()
        self.label = input[5].cuda().float()


    def forward(self):
        if self.freeze_global_branch:
            self.components = self.model(
                self.input, None, None, cla=True, return_components=True)
            self.output = None
            self.classhead = self.components['final_logits'].squeeze(1)
        else:
            self.output, self.classhead, self.components = self.model(
                self.input,
                self.input_ids,
                self.attention_mask,
                return_components=True,
            )

        
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        
        caption_loss = nn.functional.cross_entropy(logits    , torch.arange(len(logits), device=logits.device))
        image_loss   = nn.functional.cross_entropy(logits.t(), torch.arange(len(logits), device=logits.device))
        return (caption_loss + image_loss) / 2.0
        
    def optimize_parameters(self):
        self.forward()

        if self.output is None:
            self.loss1 = self.classhead.new_zeros(())
        else:
            self.loss1 = sum([
                self.contrastive_loss(output)
                for output in torch.split(
                    self.output, self.output.shape[1], dim=0)
            ])
        self.loss2 = self.claloss * self.loss_fn(self.classhead, self.label)
        self.loss_rank = self.rank_loss_weight * pairwise_ranking_loss(
            self.classhead, self.label)
        zero = self.classhead.new_zeros(())
        if 'gate' in self.components:
            self.loss_local_candidate = (
                self.local_candidate_loss_weight * residual_candidate_loss(
                    self.components['global_logits'],
                    self.components['local_logits'],
                    self.label,
                )
            )
            self.loss_preserve = (
                self.preserve_loss_weight * confidence_preservation_loss(
                    self.components['final_logits'],
                    self.components['global_logits'],
                )
            )
            self.loss_gate = self.gate_loss_weight * gate_sparsity_loss(
                self.components['gate'])
            gate_supervision, gate_targets = relative_gate_supervision_loss(
                self.components['gate'],
                self.components['global_logits'],
                self.components['local_logits'],
                self.label,
                margin=self.gate_target_margin,
            )
            self.loss_gate_supervision = (
                self.gate_supervision_weight * gate_supervision)
            self.gate_target_mean = gate_targets.detach().mean()
        else:
            self.loss_local_candidate = zero
            self.loss_preserve = zero
            self.loss_gate = zero
            self.loss_gate_supervision = zero
            self.gate_target_mean = None
        self.loss = (
            self.loss1 + self.loss2 + self.loss_rank
            + self.loss_local_candidate + self.loss_preserve
            + self.loss_gate + self.loss_gate_supervision
        )
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_local_gate_value(self):
        if hasattr(self, 'components') and 'gate' in self.components:
            return self.components['gate'].detach().mean().item()
        core_model = (
            self.model.module if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        gate = core_model.local_gate_value()
        return None if gate is None else gate.detach().item()

