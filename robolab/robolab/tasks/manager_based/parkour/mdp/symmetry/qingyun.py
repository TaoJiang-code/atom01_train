# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify left-right symmetry for QingYun Rev3 parkour training."""

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
    """Augments the given observations and actions by applying symmetry transformations.

    ``env`` is kept for compatibility with RSL-RL's symmetry callback signature.

    This function creates augmented versions of the provided observations and actions by applying
    two transformations: original and left-right mirrored. This is useful because a QingYun motion
    that is valid on the left side should usually have a valid mirrored version on the right side.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # Since we keep the original sample and one left-right mirrored sample, the batch is doubled.
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"])

        # critic observation group
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        # -- left-right
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs["critic"])

    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # Since we keep the original action and one left-right mirrored action, the batch is doubled.
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)

    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _height_scan_left_right_dims(env: ManagerBasedRLEnv) -> tuple[int, int, int]:
    """(history_length, ny, nx) from the running env cfg (dataclass fields are not on config *classes*)."""
    cfg = getattr(env, "unwrapped", env).cfg
    pat = cfg.scene.height_scanner.pattern_cfg
    if pat.ordering != "xy":
        raise NotImplementedError(
            "height_scan L-R symmetry only supports GridPatternCfg ordering 'xy';"
            f" extend layouts if pattern uses ordering {pat.ordering!r}."
        )
    hist = cfg.observations.critic.height_scan.history_length
    res = float(pat.resolution)
    s0, s1 = float(pat.size[0]), float(pat.size[1])
    # Match ``isaaclab.sensors.ray_caster.patterns.grid_pattern`` (same arange endpoints / step).
    nx = int(torch.arange(-s0 / 2, s0 / 2 + 1.0e-9, res).numel())
    ny = int(torch.arange(-s1 / 2, s1 / 2 + 1.0e-9, res).numel())
    return hist, ny, nx


def _transform_height_scan_left_right(env: ManagerBasedRLEnv, hs: torch.Tensor) -> torch.Tensor:
    """Mirror lateral (y) grid rows; scalars only reorder (unlike ``depth_image``, scan is flat w.r.t. spatial axes)."""
    hist, ny, nx = _height_scan_left_right_dims(env)
    out = hs.view(hs.shape[0], hist, ny, nx).flip(dims=[2])
    return out.reshape(hs.shape)


def _transform_policy_obs_left_right(obs: TensorDict) -> TensorDict:
    """Left-right mirror for policy observations (``ObservationsCfg.PolicyCfg`` with ``concatenate_terms=False``)."""
    obs = obs.clone()
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    if "depth_image" in obs:
        obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: TensorDict) -> TensorDict:
    """Left-right mirror for critic observations."""
    obs = obs.clone()
    obs["base_lin_vel"] = _apply_xyz_sign(obs["base_lin_vel"], [1, -1, 1])
    obs["base_ang_vel"] = _apply_xyz_sign(obs["base_ang_vel"], [-1, 1, -1])
    obs["projected_gravity"] = _apply_xyz_sign(obs["projected_gravity"], [1, -1, 1])
    obs["velocity_commands"] = _apply_xyz_sign(obs["velocity_commands"], [1, -1, -1])
    obs["joint_pos"] = _switch_joints_left_right_flat(obs["joint_pos"])
    obs["joint_vel"] = _switch_joints_left_right_flat(obs["joint_vel"])
    obs["actions"] = _switch_joints_left_right_flat(obs["actions"])
    if "depth_image" in obs:
        obs["depth_image"] = _transform_depth_obs_left_right(obs["depth_image"])
    if "height_scan" in obs:
        obs["height_scan"] = _transform_height_scan_left_right(env, obs["height_scan"])
    return obs


def _transform_depth_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to depth image observations."""
    return torch.flip(obs, dims=(-1,))


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, QINGYUN_NUM_JOINTS)
    joint_data = _switch_joints_left_right(joint_data)
    return joint_data.reshape(joint_data_shape)


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the QingYun action tensor.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Lab, the QingYun Rev3 joint ordering used by the articulation is:
[
    "lw_hip_pitch_joint",       # 0
    "rw_hip_pitch_joint",       # 1
    "waist_yaw_joint",          # 2
    "lw_hip_roll_joint",        # 3
    "rw_hip_roll_joint",        # 4
    "lw_shoulder_pitch_joint",  # 5
    "rw_shoulder_pitch_joint",  # 6
    "lw_hip_yaw_joint",         # 7
    "rw_hip_yaw_joint",         # 8
    "lw_arm_roll_joint",        # 9
    "rw_arm_roll_joint",        # 10
    "lw_knee_pitch_joint",      # 11
    "rw_knee_pitch_joint",      # 12
    "lw_arm_yaw_joint",         # 13
    "rw_arm_yaw_joint",         # 14
    "lw_foot_pitch_joint",      # 15
    "rw_foot_pitch_joint",      # 16
    "lw_elbow_pitch_joint",     # 17
    "rw_elbow_pitch_joint",     # 18
]
"""

QINGYUN_NUM_JOINTS = 19

QINGYUN_LEFT_JOINT_IDS = [0, 3, 5, 7, 9, 11, 13, 15, 17]
QINGYUN_RIGHT_JOINT_IDS = [1, 4, 6, 8, 10, 12, 14, 16, 18]

# Under a left-right mirror, QingYun's left/right paired joint coordinates change sign.
# The central waist yaw has no opposite-side partner, but its yaw direction also reverses.
QINGYUN_NEGATED_JOINT_IDS = list(range(QINGYUN_NUM_JOINTS))


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to QingYun joint data."""
    joint_data_switched = joint_data.clone()
    # left <-- right
    joint_data_switched[..., QINGYUN_LEFT_JOINT_IDS] = joint_data[..., QINGYUN_RIGHT_JOINT_IDS]
    # right <-- left
    joint_data_switched[..., QINGYUN_RIGHT_JOINT_IDS] = joint_data[..., QINGYUN_LEFT_JOINT_IDS]

    joint_data_switched[..., QINGYUN_NEGATED_JOINT_IDS] = (
        -1 * joint_data_switched[..., QINGYUN_NEGATED_JOINT_IDS]
    )

    return joint_data_switched
