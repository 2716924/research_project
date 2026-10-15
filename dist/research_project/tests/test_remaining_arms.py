"""Paper-derived target oracles and integration checks for the other arms."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rct.network import MinAtarQNetwork
from rct.targets import AveragedTargetEnsemble, control_target, double_dqn_target


class FixedQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.expand(len(states), -1)


def constant_network(values):
    net = MinAtarQNetwork(1, len(values), num_layers=1)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.output_layer.bias.copy_(torch.tensor(values))
    return net


class RemainingArmTests(unittest.TestCase):
    def test_dqn_and_ddqn_disagree_as_paper_requires(self):
        # Online prefers action 0, target prefers action 1. Using the wrong
        # network for either step changes the answer; terminal rows stay fixed.
        online, target = FixedQ([10., 1.]), FixedQ([2., 8.])
        states = torch.zeros(2, 1)
        rewards, dones = torch.tensor([[1.], [3.]]), torch.tensor([[0.], [1.]])
        dqn = control_target(states, rewards, dones, target, .9)
        ddqn = double_dqn_target(states, rewards, dones, online, target, .9)
        torch.testing.assert_close(dqn, torch.tensor([[8.2], [3.]]))
        torch.testing.assert_close(ddqn, torch.tensor([[2.8], [3.]]))
        self.assertFalse(dqn.requires_grad)
        self.assertFalse(ddqn.requires_grad)
        self.assertIsNone(online.values.grad)
        self.assertIsNone(target.values.grad)

    def test_average_then_max_not_mean_of_maxima(self):
        ensemble = AveragedTargetEnsemble(2, 1, 2, 1, torch.device('cpu'))
        first, second = constant_network([10., 0.]), constant_network([0., 10.])
        ensemble.snapshot(first)
        ensemble.snapshot(second)
        states = torch.zeros(2, 10, 10, 1)
        result = ensemble.compute_target(states, torch.tensor([[1.], [7.]]),
                                         torch.tensor([[0.], [1.]]), first, .9)
        # max(mean([10,0], [0,10])) = 5, whereas mean(max(...)) = 10.
        torch.testing.assert_close(result, torch.tensor([[5.5], [7.]]))
        self.assertFalse(result.requires_grad)

    def test_snapshot_copy_eviction_and_initial_fallback(self):
        ensemble = AveragedTargetEnsemble(2, 1, 2, 1, torch.device('cpu'))
        net = constant_network([1., 3.])
        states, rewards, dones = torch.zeros(1, 10, 10, 1), torch.zeros(1, 1), torch.zeros(1, 1)
        self.assertAlmostEqual(ensemble.compute_target(states, rewards, dones, net, 1).item(), 3.)
        for values in [[1., 3.], [4., 2.], [6., 0.]]:
            with torch.no_grad():
                net.output_layer.bias.copy_(torch.tensor(values))
            ensemble.snapshot(net)
        with torch.no_grad():
            net.output_layer.bias.fill_(1000)
        self.assertEqual(len(ensemble.snapshots), 2)
        self.assertAlmostEqual(ensemble.compute_target(states, rewards, dones, net, 1).item(), 5.)

    def test_training_dispatch_target_sync_and_all_arms_smoke(self):
        from rct import trainer
        torch.set_num_threads(1)
        individual = dict(seed=1337, config=dict(lr=.0001, buffer_capacity=100,
                          batch_size=8, num_layers=1, replay_start_size=12,
                          target_update_freq=20, epsilon_decay_steps=100))
        real_network = trainer.MinAtarQNetwork
        real_ensemble = trainer.AveragedTargetEnsemble
        for variant in ["control", "ddqn", "per", "nstep", "averaged"]:
            with self.subTest(variant=variant):
                nets, ensembles = [], []

                def network(*args, **kwargs):
                    net = real_network(*args, **kwargs)
                    nets.append(net)
                    return net

                def ensemble(*args, **kwargs):
                    item = real_ensemble(*args, **kwargs)
                    item.snapshot = unittest.mock.Mock(wraps=item.snapshot)
                    ensembles.append(item)
                    return item

                with patch.object(trainer, "MinAtarQNetwork", side_effect=network), \
                     patch.object(trainer, "AveragedTargetEnsemble", side_effect=ensemble), \
                     patch.object(trainer, "control_target", wraps=trainer.control_target) as dqn, \
                     patch.object(trainer, "double_dqn_target", wraps=trainer.double_dqn_target) as ddqn, \
                     patch.object(trainer, "nstep_target", wraps=trainer.nstep_target) as nstep, \
                     patch.object(trainer, "evaluate_agent", return_value=0.), \
                     patch.object(trainer, "should_stop_for_time_limit", return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    trajectory, completed = trainer.train_individual(
                        variant, "MinAtar/Breakout-v1", individual, 4, 3,
                        total_frames=120, eval_frequency=40, num_eval_episodes=1, log_every=40)
                self.assertTrue(completed)
                self.assertEqual(len(trajectory), 3)
                self.assertEqual(dqn.called, variant in {"control", "per"})
                self.assertEqual(ddqn.called, variant == "ddqn")
                self.assertEqual(nstep.called, variant == "nstep")
                self.assertTrue(all(p.grad is None for p in nets[1].parameters()))
                self.assertTrue(all(torch.isfinite(p).all() for p in nets[0].parameters()))
                # Frame 120 is a synchronization boundary for every arm.
                for online, target in zip(nets[0].parameters(), nets[1].parameters()):
                    torch.testing.assert_close(online, target, rtol=0, atol=0)
                if ensembles:
                    self.assertEqual(ensembles[0].snapshot.call_count, 6)
                    self.assertEqual(len(ensembles[0].snapshots), 6)


if __name__ == '__main__':
    unittest.main()
