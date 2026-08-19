"""Compatibility guard for training experiments removed from active code."""

import sys


RETIRED_TRAINING_FLAGS = frozenset({
    '--augmentation_dro_weight',
    '--balanced_bias_calibration',
    '--boundary_center_weight',
    '--classification_referenced_gradient_cap',
    '--degradation_consistency_weight',
    '--degradation_scale',
    '--ema_decay',
    '--freeze_global_branch',
    '--freeze_vision_lora',
    '--gate_loss_weight',
    '--gate_supervision_weight',
    '--gate_target_margin',
    '--global_contrastive',
    '--global_contrastive_weight',
    '--gradient_accumulation_steps',
    '--gradient_conflict_diagnostics',
    '--gradient_conflict_projection',
    '--hard_fake_semantic_coverage',
    '--init_baseline_checkpoint',
    '--local_candidate_loss_weight',
    '--local_dim',
    '--local_dropout',
    '--local_fusion',
    '--local_gate_init',
    '--local_layer',
    '--local_pool',
    '--logit_center_loss_weight',
    '--logit_margin',
    '--margin_loss_weight',
    '--paired_authenticity_head_initialization',
    '--paired_authenticity_normalize_direction',
    '--patch_residual_head',
    '--rank_loss_weight',
    '--residual_alpha',
    '--residual_scale',
    '--residual_trust_weight',
    '--residual_vib',
    '--rrsd_max_correction',
    '--semantic_residual_weight',
    '--spectral_band_dropout',
    '--symmetric_prototype_head',
    '--use_local_features',
    '--vib_beta',
    '--vib_cls_weight',
    '--vib_dim',
})


def reject_retired_training_flags(argv=None):
    """Reject removed options instead of silently running another method."""
    argv = sys.argv[1:] if argv is None else argv
    used_flags = {
        argument.split('=', 1)[0]
        for argument in argv
        if argument.startswith('--')
    }
    retired_flags = sorted(used_flags & RETIRED_TRAINING_FLAGS)
    if retired_flags:
        raise ValueError(
            'retired training options are no longer supported: '
            + ', '.join(retired_flags)
        )
