from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tracking_exp_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to increase the difficulty of the terrain when the robot tracks its commanded velocity well
    (high score). It decreases the difficulty when the robot tracks its commanded velocity poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        asset_cfg: The configuration of the robot articulation in the scene.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear velocity tracking
            score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the terrain difficulty is decreased.
            If the score is above the upper threshold (good tracking), the terrain difficulty is increased.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular velocity tracking
            score (exponential kernel).
            Similar logic applies as lin_vel_threshold.
    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def target_reaching_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    target_distance_threshold: float = 0.6,
    root_height_offset: float = 0.35,
    lin_vel_threshold: tuple = (0.7, 0.9),
) -> torch.Tensor:
    """Terrain curriculum based on reaching the sampled target patch.

    The terrain level increases when the robot reaches the command generator's sampled target
    patch and tracks the commanded linear velocity well. It decreases when the target is not
    reached or the linear velocity tracking score is too low.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term(command_name)

    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    root_pos_w = asset.data.root_pos_w[env_ids]

    if hasattr(command, "pos_command_w"):
        reference_pos_w = root_pos_w[:, :3].clone()
        reference_pos_w[:, 2] -= root_height_offset
        target_distance = torch.norm(command.pos_command_w[env_ids, :3] - reference_pos_w, dim=1)
        reached_target = target_distance < target_distance_threshold
    else:
        reached_target = torch.zeros_like(tracking_exp_vel_xy, dtype=torch.bool)

    tracking_good = tracking_exp_vel_xy > lin_vel_threshold[1]
    tracking_bad = tracking_exp_vel_xy < lin_vel_threshold[0]

    move_up = reached_target & tracking_good
    move_down = (~reached_target) | tracking_bad
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_type_tracking_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    target_distance_threshold: float = 0.6,
    root_height_reference: float = 0.35,
    upright_threshold: float = 0.5,
    min_root_height: float = 0.35,
    min_tracking_score: float = 0.35,
    fail_tracking_score: float = 0.25,
    stair_height_success_threshold: float = 0.08,
    stair_height_fail_threshold: float = 0.03,
    slope_height_success_threshold: float = 0.06,
    gap_keywords: tuple[str, ...] = ("gap",),
    stair_keywords: tuple[str, ...] = ("pyramid_stairs",),
    slope_keywords: tuple[str, ...] = ("slope",),
    fallback_lin_vel_threshold: tuple = (0.7, 0.9),
    fallback_ang_vel_threshold: tuple = (0.0, 0.0),
    fallback_to_velocity_tracking: bool = True,
) -> torch.Tensor:
    """Terrain curriculum based on terrain type, stability, tracking, and target reaching.

    Stairs and slopes can succeed either by reaching the sampled target patch or by changing
    root height enough to indicate obstacle progress. Gaps require target reaching. Unknown
    terrains optionally fall back to pure velocity-tracking curriculum.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term(command_name)

    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)

    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    root_pos_w = asset.data.root_pos_w[env_ids]
    env_origins = env.scene.env_origins[env_ids]

    upright_score = -asset.data.projected_gravity_b[env_ids, 2]
    root_height = root_pos_w[:, 2] - env_origins[:, 2]
    stable = (upright_score > upright_threshold) & (root_height > min_root_height)
    terrain_height_change = root_height - root_height_reference

    if hasattr(command, "pos_command_w"):
        target_distance = torch.norm(command.pos_command_w[env_ids, :2] - root_pos_w[:, :2], dim=1)
        reached_target = target_distance < target_distance_threshold
    else:
        reached_target = torch.zeros_like(tracking_exp_vel_xy, dtype=torch.bool)

    gap_mask = torch.zeros_like(tracking_exp_vel_xy, dtype=torch.bool)
    stair_mask = torch.zeros_like(tracking_exp_vel_xy, dtype=torch.bool)
    slope_mask = torch.zeros_like(tracking_exp_vel_xy, dtype=torch.bool)
    if terrain.cfg.terrain_type == "generator":
        terrain_gen = getattr(terrain, "terrain_generator", None)
        if terrain_gen is not None and hasattr(terrain_gen, "get_subterrain_indices"):
            sub_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
            sub_idx_per_env = terrain_gen.get_subterrain_indices(
                terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], device=env.device
            )
            for sub_idx, name in enumerate(sub_names):
                name_l = name.lower()
                sub_mask = sub_idx_per_env == sub_idx
                if any(keyword.lower() in name_l for keyword in gap_keywords):
                    gap_mask |= sub_mask
                elif any(keyword.lower() in name_l for keyword in stair_keywords):
                    stair_mask |= sub_mask
                elif any(keyword.lower() in name_l for keyword in slope_keywords):
                    slope_mask |= sub_mask

    known_terrain_mask = gap_mask | stair_mask | slope_mask
    tracking_good_enough = tracking_exp_vel_xy > min_tracking_score
    tracking_failed = tracking_exp_vel_xy < fail_tracking_score

    stair_progress = torch.abs(terrain_height_change) > stair_height_success_threshold
    stair_stuck = torch.abs(terrain_height_change) < stair_height_fail_threshold
    slope_progress = torch.abs(terrain_height_change) > slope_height_success_threshold

    stair_success = stair_mask & stable & tracking_good_enough & (reached_target | stair_progress)
    gap_success = gap_mask & stable & tracking_good_enough & reached_target
    slope_success = slope_mask & stable & tracking_good_enough & (reached_target | slope_progress)

    stair_fail = stair_mask & (~stable | (stair_stuck & tracking_failed & ~reached_target))
    gap_fail = gap_mask & (~stable | (tracking_failed & ~reached_target))
    slope_fail = slope_mask & (~stable | (tracking_failed & ~reached_target))

    move_up = stair_success | gap_success | slope_success
    move_down = stair_fail | gap_fail | slope_fail

    if fallback_to_velocity_tracking:
        unknown_mask = ~known_terrain_mask
        fallback_up = (
            unknown_mask
            & (tracking_exp_vel_xy > fallback_lin_vel_threshold[1])
            & (tracking_exp_vel_yaw > fallback_ang_vel_threshold[1])
        )
        fallback_down = unknown_mask & (tracking_exp_vel_xy < fallback_lin_vel_threshold[0])
        move_up |= fallback_up
        move_down |= fallback_down

    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())



def modify_rewards_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    init_weight: float,
    final_weight: float,
    lin_vel_threshold: tuple = (0.3, 0.6),
    ang_vel_threshold: tuple = (0.3, 0.5),
    step_size: float = 0.02,
) -> torch.Tensor:
    """Curriculum based on the velocity tracking performance (exponential score) of the robot.

    This term is used to gradually adjust a reward term's weight from ``init_weight`` toward
    ``final_weight`` when the robot tracks its commanded velocity well (high score), and step
    it back toward ``init_weight`` when the robot tracks poorly (low score).

    Args:
        env: The learning environment.
        env_ids: The environment ids for which the curriculum should be computed.
        term_name: Name of the reward term whose weight will be modified.
        init_weight: Initial (easy) weight of the reward term.
        final_weight: Final (strict) weight to ramp toward when tracking is good.
        lin_vel_threshold: A tuple specifying the lower and upper threshold for the linear
            velocity tracking score (exponential kernel).
            If the score is below the lower threshold (poor tracking), the weight is moved
            back toward ``init_weight``.
            If the score is above the upper threshold (good tracking), the weight is moved
            toward ``final_weight``.
        ang_vel_threshold: A tuple specifying the lower and upper threshold for the angular
            velocity tracking score (exponential kernel). Similar logic as ``lin_vel_threshold``.
        step_size: Fractional step taken toward the target weight each time this curriculum
            is triggered (0.02 = move 2% of the remaining gap).
        group_name: Name of the reward group that contains ``term_name``. If ``None``, the
            first group is used (matches ``MultiRewardManager`` behavior).

    Returns:
        The global mean reward term weight as a tensor scalar, for logging.
    """
    # extract the used quantities
    command = env.command_manager.get_term("base_velocity")
    tracking_exp_vel_xy = command.metrics["tracking_exp_vel_xy"][env_ids]
    tracking_exp_vel_yaw = command.metrics["tracking_exp_vel_yaw"][env_ids]
    # decide whether to ramp up (toward final_weight) or back down (toward init_weight)
    move_up = (tracking_exp_vel_xy > lin_vel_threshold[1]) * (tracking_exp_vel_yaw > ang_vel_threshold[1])
    move_down = tracking_exp_vel_xy < lin_vel_threshold[0]
    move_down *= ~move_up

    # update per-environment weights for the specified envs only
    per_env_weights = env.reward_manager.get_per_env_term_weights(term_name)
    # normalize env_ids to tensor for indexing
    if not isinstance(env_ids, torch.Tensor):
        env_idx = torch.tensor(list(env_ids), dtype=torch.long, device=per_env_weights.device)
    else:
        env_idx = env_ids.to(device=per_env_weights.device)
    current = per_env_weights[env_idx]
    # move_up and move_down are boolean tensors aligned with env_ids
    move_up = move_up.to(dtype=current.dtype, device=current.device)
    move_down = move_down.to(dtype=current.dtype, device=current.device)
    # per-env updates
    current = current + (final_weight - current) * step_size * move_up
    current = current + (init_weight - current) * step_size * move_down
    # write back only for these envs
    env.reward_manager.set_term_weight_for_envs(term_name, env_idx, current)

    # Log the global mean so the curve reflects the full population, not only the reset batch.
    global_weights = env.reward_manager.get_per_env_term_weights(term_name)
    return global_weights.mean()
