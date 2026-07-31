"""
Target-value calculations for each RCT algorithmic arm.

Each function changes only the part of DQN described in Table 4.2
of the research proposal, keeping everything else identical.
"""

import copy

import torch

from rct.network import MinAtarQNetwork


def control_target(next_states, rewards, dones, target_network, gamma):
    """Canonical 1-step DQN target — Mnih et al. 2015."""
    with torch.no_grad():
        next_q = target_network(next_states).max(1)[0].unsqueeze(1)
        return rewards + (gamma * next_q * (1 - dones))


def nstep_target(next_states, rewards, dones, target_network, gamma, n_step):
    """
    n-step DQN target — Hernandez-Garcia & Sutton 2019.

    Transitions store the discounted n-step return sum; bootstrap with
    gamma^n on the state reached after n steps (or zero if terminal).
    """
    with torch.no_grad():
        next_q = target_network(next_states).max(1)[0].unsqueeze(1)
        bootstrap = (gamma ** n_step) * next_q * (1 - dones)
        return rewards + bootstrap


def double_dqn_target(next_states, rewards, dones, q_network, target_network, gamma):
    """
    Double DQN target — Van Hasselt et al. 2016.

    Online network picks the action; target network scores it.
    This reduces the overestimation bias of vanilla DQN.
    """
    with torch.no_grad():
        online_q = q_network(next_states)
        best_actions = online_q.argmax(dim=1, keepdim=True)
        next_q = target_network(next_states).gather(1, best_actions)
        return rewards + (gamma * next_q * (1 - dones))


class AveragedTargetEnsemble:
    """
    Averaged DQN target ensemble — Anschel et al. 2017.

    Keeps K snapshots of past target networks and averages their
    max-Q estimates to produce a smoother, lower-variance target.
    """

    def __init__(self, ensemble_size, in_channels, num_actions, num_layers, device):
        self.ensemble_size = ensemble_size
        self.device = device
        self.snapshots = []
        self._eval_network = MinAtarQNetwork(in_channels, num_actions, num_layers=num_layers).to(device)
        self._eval_network.eval()

    def snapshot(self, target_network):
        state_dict = copy.deepcopy(target_network.state_dict())
        if len(self.snapshots) < self.ensemble_size:
            self.snapshots.append(state_dict)
        else:
            self.snapshots.pop(0)
            self.snapshots.append(state_dict)

    def compute_target(self, next_states, rewards, dones, fallback_target_network, gamma):
        with torch.no_grad():
            if not self.snapshots:
                next_q = fallback_target_network(next_states).max(1)[0].unsqueeze(1)
            else:
                q_sum = None
                for state_dict in self.snapshots:
                    self._eval_network.load_state_dict(state_dict)
                    q_all = self._eval_network(next_states)
                    q_sum = q_all if q_sum is None else q_sum + q_all
                avg_q = q_sum / len(self.snapshots)
                next_q = avg_q.max(1)[0].unsqueeze(1)
            return rewards + (gamma * next_q * (1 - dones))
