
"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
        
        # critic observation group
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        # -- left-right
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env.unwrapped, obs["critic"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)

    else:
        actions_aug = None

    return obs_aug, actions_aug

def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # ang vel
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 9:36] = _switch_joints_left_right(obs[:, 9:36])
    # joint vel
    obs[:, 36:63] = _switch_joints_left_right(obs[:, 36:63])
    # last actions
    obs[:, 63:90] = _switch_joints_left_right(obs[:, 63:90])

    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:

    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1, -1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 12:39] = _switch_joints_left_right(obs[:, 12:39])
    # joint vel
    obs[:, 39:66] = _switch_joints_left_right(obs[:, 39:66])
    # last actions
    obs[:, 66:93] = _switch_joints_left_right(obs[:, 66:93])

    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions

def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    act_mirror_indices = [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25]
    act_mirror_signs = [1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    joint_data_switched = joint_data.clone()
    joint_data_switched = joint_data[..., act_mirror_indices]
    joint_data_switched = joint_data_switched * act_mirror_signs

    return joint_data_switched