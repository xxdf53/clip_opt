"""Patchwise label-discriminative initialization for vision LoRA layers."""

import math
from dataclasses import dataclass

import torch


LORA_PROJECTIONS = ('q_proj', 'k_proj', 'v_proj')
DEFAULT_A_ROW_NORM = 1.0 / math.sqrt(3.0)


@dataclass(frozen=True)
class PLDInitializationSummary:
    layers: int
    modules: int
    real_samples: int
    fake_samples: int
    rank: int
    explained_energy: float


def _attention_lora_groups(model):
    modules = dict(model.named_modules())
    groups = []
    for name, module in modules.items():
        if name.rsplit('.', 1)[-1] != 'q_proj':
            continue
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue

        prefix = name.rsplit('.', 1)[0] if '.' in name else ''
        group = []
        for projection in LORA_PROJECTIONS:
            module_name = f'{prefix}.{projection}' if prefix else projection
            projection_module = modules.get(module_name)
            if projection_module is None:
                raise ValueError(
                    f'PLD-LoRA could not find {module_name} beside {name}')
            if not hasattr(projection_module, 'lora_A'):
                raise TypeError(f'{module_name} is not a LoRA projection')
            group.append((module_name, projection_module))
        groups.append((name, module, group))

    if not groups:
        raise ValueError('PLD-LoRA found no q/k/v LoRA attention groups')
    return groups


def _single_adapter(module, module_name):
    adapter_names = list(module.lora_A.keys())
    if len(adapter_names) != 1:
        raise ValueError(
            f'PLD-LoRA requires one adapter in {module_name}; '
            f'found {adapter_names}')
    adapter_name = adapter_names[0]
    if adapter_name not in module.lora_B:
        raise ValueError(f'{module_name} has no matching LoRA B adapter')
    return adapter_name


class _ClassActivationCollector:
    def __init__(self, layer_names):
        self.labels = None
        self.sums = {
            name: {'real': None, 'fake': None}
            for name in layer_names
        }
        self.counts = {'real': 0, 'fake': 0}

    def set_labels(self, labels):
        labels = labels.flatten()
        self.labels = labels
        self.counts['real'] += int((labels < 0.5).sum().item())
        self.counts['fake'] += int((labels >= 0.5).sum().item())

    def hook(self, layer_name):
        def collect(_, inputs):
            activations = inputs[0]
            if activations.ndim != 3:
                raise ValueError(
                    f'{layer_name} input must have shape [batch, tokens, dim]')
            if self.labels is None or activations.shape[0] != self.labels.numel():
                raise ValueError('PLD-LoRA labels do not match activations')

            for class_name, mask in (
                ('real', self.labels < 0.5),
                ('fake', self.labels >= 0.5),
            ):
                if not mask.any():
                    continue
                class_sum = (
                    activations[mask]
                    .detach()
                    .float()
                    .sum(dim=0)
                    .cpu()
                )
                previous = self.sums[layer_name][class_name]
                self.sums[layer_name][class_name] = (
                    class_sum if previous is None else previous + class_sum)

        return collect


def _canonicalize_rows(rows):
    largest_indices = rows.abs().argmax(dim=1, keepdim=True)
    signs = rows.gather(1, largest_indices).sign()
    signs[signs == 0] = 1
    return rows * signs


def _discriminant_basis(real_sum, fake_sum, real_count, fake_count, rank):
    difference = fake_sum / fake_count - real_sum / real_count
    if difference.shape[0] > 1:
        difference = difference[1:]
    if rank > min(difference.shape):
        raise ValueError(
            f'LoRA rank {rank} exceeds PLD matrix shape '
            f'{tuple(difference.shape)}')

    _, singular_values, right_vectors = torch.linalg.svd(
        difference,
        full_matrices=False,
    )
    selected = singular_values[:rank]
    if selected[-1] <= torch.finfo(selected.dtype).eps:
        raise ValueError('PLD-LoRA class-difference matrix has insufficient rank')
    basis = _canonicalize_rows(right_vectors[:rank])
    total_energy = singular_values.square().sum().clamp_min(
        torch.finfo(singular_values.dtype).eps)
    explained_energy = selected.square().sum() / total_energy
    return basis, explained_energy.item()


@torch.no_grad()
def initialize_patchwise_discriminant_lora(
    model,
    images,
    labels,
    forward_images,
    microbatch_size=8,
):
    """Initialize q/k/v LoRA A rows from one labeled image batch.

    A layer's token-wise fake-minus-real activation matrix supplies its top
    right singular vectors. LoRA B remains zero, preserving the pretrained
    model function before the first optimizer step.
    """
    if microbatch_size <= 0:
        raise ValueError('PLD-LoRA microbatch size must be positive')
    if images.shape[0] != labels.numel():
        raise ValueError('PLD-LoRA images and labels must have equal batch size')
    labels = labels.flatten()
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('PLD-LoRA labels must be binary values 0 or 1')
    if not torch.any(labels == 0) or not torch.any(labels == 1):
        raise ValueError('PLD-LoRA calibration batch must contain real and fake')

    groups = _attention_lora_groups(model)
    collector = _ClassActivationCollector([name for name, _, _ in groups])
    handles = [
        module.register_forward_pre_hook(collector.hook(name))
        for name, module, _ in groups
    ]
    first_module = groups[0][1]
    first_adapter = _single_adapter(first_module, groups[0][0])
    device = first_module.lora_A[first_adapter].weight.device

    try:
        for start in range(0, images.shape[0], microbatch_size):
            stop = min(start + microbatch_size, images.shape[0])
            micro_images = images[start:stop].to(device, non_blocking=True)
            micro_labels = labels[start:stop].to(device, non_blocking=True)
            collector.set_labels(micro_labels)
            forward_images(micro_images)
    finally:
        for handle in handles:
            handle.remove()

    real_count = collector.counts['real']
    fake_count = collector.counts['fake']
    explained_energies = []
    rank = None
    module_count = 0
    for layer_name, _, projections in groups:
        sums = collector.sums[layer_name]
        if sums['real'] is None or sums['fake'] is None:
            raise RuntimeError(
                f'PLD-LoRA did not collect both classes at {layer_name}')

        first_projection_name, first_projection = projections[0]
        adapter_name = _single_adapter(
            first_projection, first_projection_name)
        rank = first_projection.lora_A[adapter_name].weight.shape[0]
        basis, explained_energy = _discriminant_basis(
            sums['real'],
            sums['fake'],
            real_count,
            fake_count,
            rank,
        )
        explained_energies.append(explained_energy)

        for module_name, projection in projections:
            projection_adapter = _single_adapter(projection, module_name)
            lora_a = projection.lora_A[projection_adapter].weight
            lora_b = projection.lora_B[projection_adapter].weight
            if lora_a.shape != basis.shape:
                raise ValueError(
                    f'{module_name} LoRA A shape {tuple(lora_a.shape)} does '
                    f'not match PLD basis {tuple(basis.shape)}')
            lora_a.copy_(
                basis.to(device=lora_a.device, dtype=lora_a.dtype)
                * DEFAULT_A_ROW_NORM
            )
            lora_b.zero_()
            module_count += 1

    return PLDInitializationSummary(
        layers=len(groups),
        modules=module_count,
        real_samples=real_count,
        fake_samples=fake_count,
        rank=rank,
        explained_energy=sum(explained_energies) / len(explained_energies),
    )
