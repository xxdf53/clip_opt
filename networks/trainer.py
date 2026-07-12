import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel

from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model



class CLIPModel_lora(nn.Module):
    def __init__(self, name='openai/clip-vit-large-patch14-336', num_classes=1,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 use_local_features=False, local_layer=12, local_dim=256,
                 local_dropout=0.1, local_pool='mean_std',
                 freeze_vision_lora=False):
        super(CLIPModel_lora, self).__init__()
        self.model        = CLIPModel.from_pretrained(name)
        self.processor    = CLIPProcessor.from_pretrained(name)
        self.vision_tower = self.model.vision_model
        self.vision_tower.requires_grad_(False)
        self.model.text_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.text_projection.requires_grad_(False)
        self.contrastive_loss = nn.CrossEntropyLoss()
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

            pooled_dim = vision_dim if self.local_pool == 'mean' else 2 * vision_dim
            self.local_norm = nn.LayerNorm(vision_dim)
            self.local_projector = nn.Sequential(
                nn.Linear(pooled_dim, local_dim),
                nn.GELU(),
                nn.Dropout(local_dropout),
            )
            self.model.fc = nn.Sequential(
                nn.Linear(projection_dim + local_dim, 256),
                nn.GELU(),
                nn.Dropout(local_dropout),
                nn.Linear(256, num_classes),
            )
        else:
            self.model.fc = nn.Linear(projection_dim, num_classes)

        self.model.fc.apply(self._init_linear)
        if self.use_local_features:
            self.local_projector.apply(self._init_linear)

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

    def classify(self, image_features, local_features):
        image_features = F.normalize(image_features, p=2, dim=-1)
        if not self.use_local_features:
            return image_features, self.model.fc(image_features)

        local_features = F.normalize(local_features, p=2, dim=-1)
        fused_features = torch.cat((image_features, local_features), dim=-1)
        return image_features, self.model.fc(fused_features)
    
    def forward(self, img, input_ids, attention_mask, cla=False):
        # tmp = x; print(f'x: {tmp.shape}, max: {tmp.max()}, min: {tmp.min()}, mean: {tmp.mean()}')

        image_features, local_features = self.encode_image(img)
        image_embeds, classhead = self.classify(image_features, local_features)
        if cla: return classhead 
        
        text_embeds  = self.encode_text(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)


        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.model.logit_scale.exp()
        logits_per_image = logits_per_text.t()

        return logits_per_image, classhead.squeeze(1)
    
    def forward_eval(self, img):
        image_features, local_features = self.encode_image(img)
        _, classhead = self.classify(image_features, local_features)
        return classhead 

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        self.delr = opt.delr
        self.claloss = opt.claloss
        
        self.printOne = 1
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
            freeze_vision_lora=opt.freeze_vision_lora,
        )

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
        self.text            = input[2]
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
        self.output, self.classhead = self.model( self.input, self.input_ids, self.attention_mask )

        
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        
        caption_loss = nn.functional.cross_entropy(logits    , torch.arange(len(logits), device=logits.device))
        image_loss   = nn.functional.cross_entropy(logits.t(), torch.arange(len(logits), device=logits.device))
        return (caption_loss + image_loss) / 2.0
        
    def get_loss(self):
        return self.model.clip_loss_input(self.input, self.text)

    def optimize_parameters(self):
        self.forward()
        
        self.loss1 = sum([self.contrastive_loss(output) for output in torch.split(self.output, self.output.shape[1], dim=0)])
        self.loss2 = self.claloss * self.loss_fn(self.classhead, self.label)
        self.loss  = self.loss1 + self.loss2
        # self.loss1, self.loss2 = 0.0, 0.0
        # self.loss  = self.loss_fn(self.classhead, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

