"""Training-only HFR/HRR routing from a labeled development split."""

import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.binary_evaluation import build_group_dataset, build_transform
from utils.training_objectives import (
    fake_reweighting_loss,
    hard_real_reweighting_loss,
)


ROUTES = ('hfr', 'hrr')


@dataclass(frozen=True)
class RoutingDevMetrics:
    fake_mean_bce: float | None
    real_mean_bce: float | None
    fake_count: int
    real_count: int
    elapsed_seconds: float
    reason: str

    @property
    def valid(self):
        return self.reason == 'ok'


class ValidationGuidedHardRouter:
    """Persisted anti-oscillation state for labeled routing-dev updates."""

    state_version = 1

    def __init__(
        self,
        initial_route='hfr',
        ema_decay=0.9,
        deadband=0.0,
        persistence=2,
    ):
        if initial_route not in ROUTES:
            raise ValueError(f'initial_route must be one of {ROUTES}')
        if not 0 <= ema_decay < 1:
            raise ValueError('ema_decay must be in [0, 1)')
        if deadband < 0:
            raise ValueError('deadband cannot be negative')
        if persistence <= 0:
            raise ValueError('persistence must be positive')
        self.initial_route = initial_route
        self.ema_decay = float(ema_decay)
        self.deadband = float(deadband)
        self.persistence = int(persistence)
        self.reset()

    def reset(self):
        self.current_route = self.initial_route
        self.ema_score = None
        self.pending_candidate = None
        self.candidate_streak = 0
        self.switch_count = 0
        self.last_update_step = 0

    def state_dict(self):
        return {
            'version': self.state_version,
            'initial_route': self.initial_route,
            'ema_decay': self.ema_decay,
            'deadband': self.deadband,
            'persistence': self.persistence,
            'current_route': self.current_route,
            'ema_score': self.ema_score,
            'pending_candidate': self.pending_candidate,
            'candidate_streak': self.candidate_streak,
            'switch_count': self.switch_count,
            'last_update_step': self.last_update_step,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError('routing controller state must be a dictionary')
        if state.get('version') != self.state_version:
            raise ValueError(
                'unsupported or missing routing controller state version')
        route = state.get('current_route')
        candidate = state.get('pending_candidate')
        ema_score = state.get('ema_score')
        streak = state.get('candidate_streak')
        switches = state.get('switch_count')
        last_update = state.get('last_update_step')
        saved_configuration = (
            state.get('initial_route'),
            state.get('ema_decay'),
            state.get('deadband'),
            state.get('persistence'),
        )
        current_configuration = (
            self.initial_route,
            self.ema_decay,
            self.deadband,
            self.persistence,
        )
        if saved_configuration != current_configuration:
            raise ValueError(
                'routing controller configuration does not match checkpoint')
        if route not in ROUTES:
            raise ValueError('routing state has an invalid current_route')
        if candidate not in (None, *ROUTES):
            raise ValueError('routing state has an invalid pending_candidate')
        if ema_score is not None and not math.isfinite(float(ema_score)):
            raise ValueError('routing state has a non-finite ema_score')
        if any(
            not isinstance(value, int) or value < 0
            for value in (streak, switches, last_update)
        ):
            raise ValueError('routing counters must be non-negative integers')
        self.current_route = route
        self.ema_score = (
            None if ema_score is None else float(ema_score))
        self.pending_candidate = candidate
        self.candidate_streak = streak
        self.switch_count = switches
        self.last_update_step = last_update

    def update(self, metrics, step):
        """Update route after an optimizer step for the next interval."""
        if step <= 0:
            raise ValueError('routing update step must be positive')
        self.last_update_step = int(step)
        raw_score = None
        candidate = None
        switched = False

        if metrics.valid:
            raw_score = metrics.fake_mean_bce - metrics.real_mean_bce
            if not math.isfinite(raw_score):
                reason = 'nonfinite_statistics'
            else:
                reason = 'updated'
                if self.ema_score is None:
                    self.ema_score = float(raw_score)
                else:
                    self.ema_score = (
                        self.ema_decay * self.ema_score
                        + (1.0 - self.ema_decay) * raw_score
                    )
                if self.ema_score > self.deadband:
                    candidate = 'hfr'
                elif self.ema_score < -self.deadband:
                    candidate = 'hrr'
                else:
                    reason = 'deadband'
        else:
            reason = metrics.reason

        if candidate is None or candidate == self.current_route:
            self.pending_candidate = None
            self.candidate_streak = 0
        elif candidate == self.pending_candidate:
            self.candidate_streak += 1
        else:
            self.pending_candidate = candidate
            self.candidate_streak = 1

        if (
            candidate is not None
            and candidate != self.current_route
            and self.candidate_streak >= self.persistence
        ):
            self.current_route = candidate
            self.switch_count += 1
            self.pending_candidate = None
            self.candidate_streak = 0
            switched = True

        return {
            'update_step': int(step),
            'fake_mean_bce': metrics.fake_mean_bce,
            'real_mean_bce': metrics.real_mean_bce,
            'fake_count': metrics.fake_count,
            'real_count': metrics.real_count,
            'raw_score': raw_score,
            'ema_score': self.ema_score,
            'candidate_route': candidate,
            'current_route': self.current_route,
            'candidate_streak': self.candidate_streak,
            'switch_count': self.switch_count,
            'switched': switched,
            'reason': reason,
            'eval_seconds': metrics.elapsed_seconds,
        }


def build_routing_dev_loader(
    root,
    crop_size=224,
    batch_size=64,
    num_workers=4,
    device=None,
):
    """Build a deterministic direct 0_real/1_fake routing-dev loader."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'routing-development root not found: {root}')
    if batch_size <= 0:
        raise ValueError('routing-development batch_size must be positive')
    if num_workers < 0:
        raise ValueError('routing-development num_workers cannot be negative')
    dataset = build_group_dataset([root], build_transform(crop_size))
    device = device or torch.device('cpu')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
    )


def evaluate_routing_dev(model, loader, device):
    """Measure class-specific mean BCE without gradients or mode leakage."""
    was_training = model.training
    start = time.perf_counter()
    fake_sum = 0.0
    real_sum = 0.0
    fake_count = 0
    real_count = 0
    model.eval()
    try:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            start = time.perf_counter()
        with torch.no_grad():
            for images, labels, _paths in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float().flatten()
                logits = model(images, cla=True).flatten()
                if logits.numel() != labels.numel():
                    raise ValueError(
                        'routing model output size must match labels')
                losses = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction='none')
                fake_mask = labels == 1
                real_mask = labels == 0
                if fake_mask.any():
                    fake_sum += losses[fake_mask].double().sum().item()
                    fake_count += int(fake_mask.sum().item())
                if real_mask.any():
                    real_sum += losses[real_mask].double().sum().item()
                    real_count += int(real_mask.sum().item())
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        model.train(was_training)

    if fake_count == 0 and real_count == 0:
        return RoutingDevMetrics(
            None, None, 0, 0, elapsed, 'empty_loader')
    if fake_count == 0:
        return RoutingDevMetrics(
            None, real_sum / real_count, 0, real_count, elapsed,
            'missing_fake')
    if real_count == 0:
        return RoutingDevMetrics(
            fake_sum / fake_count, None, fake_count, 0, elapsed,
            'missing_real')
    fake_mean = fake_sum / fake_count
    real_mean = real_sum / real_count
    if not math.isfinite(fake_mean) or not math.isfinite(real_mean):
        return RoutingDevMetrics(
            None, None, fake_count, real_count, elapsed,
            'nonfinite_statistics')
    return RoutingDevMetrics(
        fake_mean, real_mean, fake_count, real_count, elapsed, 'ok')


def routed_hard_loss(
    logits,
    labels,
    route,
    fake_fraction=0.25,
    real_fraction=0.25,
):
    """Return one existing hard-side budget and selected-count diagnostics."""
    zero = logits.new_zeros(())
    diagnostics = {
        'routing_hard_fake_selected': zero,
        'routing_hard_real_selected': zero,
    }
    if route == 'hfr':
        loss, selected = fake_reweighting_loss(
            logits, labels, fraction=fake_fraction, mode='hard')
        diagnostics['routing_hard_fake_selected'] = selected[
            'hard_fake_selected']
        return loss, diagnostics
    if route == 'hrr':
        loss, selected = hard_real_reweighting_loss(
            logits, labels, fraction=real_fraction)
        diagnostics['routing_hard_real_selected'] = selected[
            'hard_real_selected']
        return loss, diagnostics
    raise ValueError(f'route must be one of {ROUTES}')


def restore_router_state(router, training_state):
    """Restore exact state or deterministically keep the configured initial route."""
    payload = None
    if isinstance(training_state, dict):
        payload = training_state.get('validation_guided_router')
    if payload is None:
        warnings.warn(
            'checkpoint has no validation-guided routing state; '
            'using the configured deterministic initial route',
            RuntimeWarning,
        )
        router.reset()
        return False
    router.load_state_dict(payload)
    return True


def should_update_route(step, total_steps, interval):
    """Schedule post-step updates only when a subsequent interval exists."""
    return 0 < step < total_steps and step % interval == 0


def format_optional(value):
    return 'na' if value is None else f'{value:.6f}'


def format_routing_update(diagnostics):
    """Format one auditable routing-development update record."""
    candidate = diagnostics['candidate_route'] or 'hold'
    return (
        f"routing_update_step={diagnostics['update_step']} "
        f"routing_fake_bce={format_optional(diagnostics['fake_mean_bce'])} "
        f"routing_real_bce={format_optional(diagnostics['real_mean_bce'])} "
        f"routing_raw_score={format_optional(diagnostics['raw_score'])} "
        f"routing_ema_score={format_optional(diagnostics['ema_score'])} "
        f"routing_candidate={candidate} "
        f"routing_current={diagnostics['current_route']} "
        f"routing_streak={diagnostics['candidate_streak']} "
        f"routing_switch_count={diagnostics['switch_count']} "
        f"routing_switched={int(diagnostics['switched'])} "
        f"routing_fake_count={diagnostics['fake_count']} "
        f"routing_real_count={diagnostics['real_count']} "
        f"routing_eval_seconds={diagnostics['eval_seconds']:.6f} "
        f"routing_reason={diagnostics['reason']}"
    )
