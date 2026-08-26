import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from networks.base_model import BaseModel
from networks.trainer import Trainer
from utils.training_objectives import (
    fake_reweighting_loss,
    hard_real_reweighting_loss,
)
from utils.validation_routing import (
    RoutingDevMetrics,
    ValidationGuidedHardRouter,
    build_routing_dev_loader,
    evaluate_routing_dev,
    format_routing_update,
    restore_router_state,
    routed_hard_loss,
    should_update_route,
)


def metrics(fake=2.0, real=1.0, reason='ok'):
    return RoutingDevMetrics(fake, real, 2, 2, 0.25, reason)


class RecordingRoutingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_modes = []

    def forward(self, images, cla=False):
        self.grad_modes.append(torch.is_grad_enabled())
        if not cla:
            raise AssertionError('routing evaluation must be image-only cla')
        return images[:, 0, 0, 0].unsqueeze(1)


class ValidationRoutingTests(unittest.TestCase):
    def test_loader_accepts_direct_binary_layout_without_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_name, color in (
                ('0_real', (0, 0, 0)),
                ('1_fake', (255, 255, 255)),
            ):
                class_root = root / class_name
                class_root.mkdir()
                Image.new('RGB', (8, 8), color).save(class_root / 'x.png')

            loader = build_routing_dev_loader(
                root,
                crop_size=8,
                batch_size=2,
                num_workers=0,
            )
            images, labels, paths = next(iter(loader))

            self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(len(paths), 2)

    def test_score_direction_deadband_and_persistence(self):
        router = ValidationGuidedHardRouter(
            initial_route='hrr', deadband=0.1, persistence=2)

        first = router.update(metrics(fake=2.0, real=1.0), step=10)
        self.assertEqual(first['candidate_route'], 'hfr')
        self.assertEqual(first['current_route'], 'hrr')
        self.assertEqual(first['candidate_streak'], 1)

        second = router.update(metrics(fake=2.0, real=1.0), step=20)
        self.assertEqual(second['current_route'], 'hfr')
        self.assertTrue(second['switched'])
        self.assertEqual(second['switch_count'], 1)

        deadband_router = ValidationGuidedHardRouter(
            initial_route='hfr', deadband=0.1)
        held = deadband_router.update(
            metrics(fake=1.04, real=1.0), step=30)
        self.assertEqual(held['reason'], 'deadband')
        self.assertEqual(held['current_route'], 'hfr')

    def test_negative_score_selects_hrr(self):
        router = ValidationGuidedHardRouter(
            initial_route='hfr', persistence=1)
        result = router.update(metrics(fake=1.0, real=2.0), step=10)
        self.assertEqual(result['candidate_route'], 'hrr')
        self.assertEqual(result['current_route'], 'hrr')

    def test_ema_uses_fake_minus_real_score(self):
        router = ValidationGuidedHardRouter(ema_decay=0.5)
        router.update(metrics(fake=3.0, real=1.0), step=10)
        result = router.update(metrics(fake=1.0, real=1.0), step=20)
        self.assertAlmostEqual(result['raw_score'], 0.0)
        self.assertAlmostEqual(result['ema_score'], 1.0)

    def test_invalid_metrics_hold_route_without_nonfinite_state(self):
        router = ValidationGuidedHardRouter(initial_route='hrr')
        cases = [
            RoutingDevMetrics(None, None, 0, 0, 0.1, 'empty_loader'),
            RoutingDevMetrics(None, 1.0, 0, 2, 0.1, 'missing_fake'),
            RoutingDevMetrics(1.0, None, 2, 0, 0.1, 'missing_real'),
            RoutingDevMetrics(None, None, 2, 2, 0.1,
                              'nonfinite_statistics'),
        ]
        for step, invalid in enumerate(cases, start=1):
            result = router.update(invalid, step=step)
            self.assertEqual(result['current_route'], 'hrr')
            self.assertIsNone(result['raw_score'])
            self.assertIsNone(result['ema_score'])

    def test_next_interval_semantics_and_schedule(self):
        router = ValidationGuidedHardRouter(
            initial_route='hfr', persistence=1)
        route_used_for_current_step = router.current_route
        router.update(metrics(fake=1.0, real=2.0), step=10)

        self.assertEqual(route_used_for_current_step, 'hfr')
        self.assertEqual(router.current_route, 'hrr')
        self.assertTrue(should_update_route(10, total_steps=20, interval=10))
        self.assertFalse(should_update_route(20, total_steps=20, interval=10))

    def test_routed_loss_reuses_one_static_budget(self):
        logits = torch.tensor([-2.0, -1.0, 0.5, 1.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        expected_fake, _ = fake_reweighting_loss(
            logits, labels, fraction=0.5, mode='hard')
        expected_real, _ = hard_real_reweighting_loss(
            logits, labels, fraction=0.5)

        fake_loss, fake_diag = routed_hard_loss(
            logits, labels, 'hfr', 0.5, 0.5)
        real_loss, real_diag = routed_hard_loss(
            logits, labels, 'hrr', 0.5, 0.5)

        self.assertTrue(torch.allclose(fake_loss, expected_fake))
        self.assertTrue(torch.allclose(real_loss, expected_real))
        self.assertEqual(fake_diag['routing_hard_fake_selected'].item(), 1)
        self.assertEqual(fake_diag['routing_hard_real_selected'].item(), 0)
        self.assertEqual(real_diag['routing_hard_fake_selected'].item(), 0)
        self.assertEqual(real_diag['routing_hard_real_selected'].item(), 1)

    def test_evaluation_is_no_grad_and_restores_mode(self):
        dataset = [
            (torch.tensor([[[0.0]]]), torch.tensor(0), 'real.png'),
            (torch.tensor([[[1.0]]]), torch.tensor(1), 'fake.png'),
        ]
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = RecordingRoutingModel()
        model.train()

        result = evaluate_routing_dev(model, loader, torch.device('cpu'))

        self.assertTrue(result.valid)
        self.assertEqual(result.real_count, 1)
        self.assertEqual(result.fake_count, 1)
        self.assertTrue(model.training)
        self.assertEqual(model.grad_modes, [False])

        model.eval()
        evaluate_routing_dev(model, loader, torch.device('cpu'))
        self.assertFalse(model.training)

    def test_controller_state_roundtrip_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            opt = SimpleNamespace(
                isTrain=True,
                lr=0.1,
                checkpoints_dir=directory,
                name='routing',
                gpu_ids=[],
                new_optim=True,
            )
            save_dir = Path(directory) / 'routing'
            save_dir.mkdir()
            source = Trainer.__new__(Trainer)
            BaseModel.__init__(source, opt)
            source.model = nn.Linear(1, 1)
            source.routing_dev_enabled = True
            source.routing_dev_controller = ValidationGuidedHardRouter(
                initial_route='hfr', persistence=1)
            source.routing_dev_controller.update(
                metrics(fake=1.0, real=2.0), step=10)
            source.total_steps = 10
            source.save_networks('routing')

            payload = torch.load(
                save_dir / 'model_epoch_routing.pth',
                map_location='cpu',
                weights_only=False,
            )
            self.assertIn('validation_guided_router',
                          payload['training_state'])

            restored = Trainer.__new__(Trainer)
            BaseModel.__init__(restored, opt)
            restored.model = nn.Linear(1, 1)
            restored.routing_dev_enabled = True
            restored.routing_dev_controller = ValidationGuidedHardRouter(
                initial_route='hfr', persistence=1)
            restored.load_networks('routing')
            self.assertEqual(
                restored.routing_dev_controller.state_dict(),
                source.routing_dev_controller.state_dict(),
            )

            legacy = {
                'model': source.model.state_dict(),
                'total_steps': 3,
            }
            torch.save(legacy, save_dir / 'model_epoch_legacy.pth')
            restored.routing_dev_controller.current_route = 'hrr'
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                restored.load_networks('legacy')
            self.assertEqual(
                restored.routing_dev_controller.current_route, 'hfr')
            self.assertTrue(any('no validation-guided routing state'
                                in str(item.message) for item in caught))

    def test_state_rejects_nonfinite_values(self):
        router = ValidationGuidedHardRouter()
        state = router.state_dict()
        state['ema_score'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            router.load_state_dict(state)

    def test_state_rejects_mismatched_controller_configuration(self):
        source = ValidationGuidedHardRouter(persistence=1)
        restored = ValidationGuidedHardRouter(persistence=2)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            restored.load_state_dict(source.state_dict())

    def test_state_rejects_unknown_or_missing_version(self):
        router = ValidationGuidedHardRouter()
        unknown = router.state_dict()
        unknown['version'] = 99
        with self.assertRaisesRegex(ValueError, 'state version'):
            router.load_state_dict(unknown)

        missing = router.state_dict()
        del missing['version']
        with self.assertRaisesRegex(ValueError, 'state version'):
            router.load_state_dict(missing)

    def test_update_log_contains_audit_fields(self):
        router = ValidationGuidedHardRouter()
        text = format_routing_update(
            router.update(metrics(), step=10))
        for field in (
            'routing_update_step=10',
            'routing_fake_bce=',
            'routing_real_bce=',
            'routing_raw_score=',
            'routing_ema_score=',
            'routing_candidate=',
            'routing_current=',
            'routing_streak=',
            'routing_switch_count=',
            'routing_eval_seconds=',
            'routing_reason=',
        ):
            self.assertIn(field, text)

    def test_restore_helper_warns_and_resets_legacy_state(self):
        router = ValidationGuidedHardRouter(initial_route='hrr')
        router.current_route = 'hfr'
        with self.assertWarnsRegex(RuntimeWarning, 'no validation-guided'):
            restored = restore_router_state(router, {})
        self.assertFalse(restored)
        self.assertEqual(router.current_route, 'hrr')


if __name__ == '__main__':
    unittest.main()
