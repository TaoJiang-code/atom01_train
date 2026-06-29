import copy
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.qingyun_z1_A_rev_3_0 import QINGYUN_Z1_A_REV_3_0_19_DOF_CFG
from robolab.sensors import Grid3dPointsGeneratorCfg, get_link_prim_targets
import robolab.tasks.manager_based.parkour.mdp as mdp
from robolab.tasks.manager_based.parkour.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg

AMP_NUM_STEPS = 3
QINGYUN_ROOT_HEIGHT_MIN = 0.35

# Tune QingYun foot/knee volume-point grids here.
# These points are attached to foot/knee links and used by volume_points_penetration rewards.
QINGYUN_LEG_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.05,
    x_max=0.05,
    x_num=19,
    y_min=-0.03,
    y_max=0.03,
    y_num=7,
    z_min=-0.04,
    z_max=-0.02,
    z_num=3,
)
QINGYUN_KNEE_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.03,
    x_max=0.04,
    x_num=8,
    y_min=-0.03,
    y_max=0.03,
    y_num=7,
    z_min=-0.3,
    z_max=0.0,
    z_num=31,
)

# Tune QingYun command sampling here.
# Format: (min, max), units are m/s for lin_vel_x/y and rad/s for ang_vel_z.
QINGYUN_COMMAND_RANGES = {
    "lin_vel_x": (0.0, 0.0),
    "lin_vel_y": (0.0, 0.0),
    "ang_vel_z": (-1.0, 1.0),
}
QINGYUN_TERRAIN_VELOCITY_RANGES = {
    "perlin_rough": {"lin_vel_x": (0.4, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "perlin_rough_walk": {"lin_vel_x": (0.4, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
    "perlin_rough_trun": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
    "square_gaps": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "pyramid_stairs_32": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "pyramid_stairs_30": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "pyramid_stairs_28": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "pyramid_stairs_inv_32": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "pyramid_stairs_inv_30": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "pyramid_stairs_inv_28": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
    "hf_pyramid_slope_inv": {"lin_vel_x": (0.4, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
}
QINGYUN_RANDOM_VELOCITY_TERRAIN = ["perlin_rough_stand"]
QINGYUN_COMMAND_RESAMPLING_TIME_RANGE = (8.0, 12.0)
QINGYUN_REL_STANDING_ENVS = 0.05
QINGYUN_STRAIGHT_TARGET_PROB = 0.8
QINGYUN_ONLY_POSITIVE_LIN_VEL_X = True
QINGYUN_LIN_VEL_THRESHOLD = 0.0
QINGYUN_ANG_VEL_THRESHOLD = 0.0

# Tune QingYun reward weights here.
# Set a weight to 0.0 to keep the term active but make it contribute no reward.
QINGYUN_REWARD_WEIGHTS = {
    # Task rewards
    "track_lin_vel_xy_exp": 5.0,
    "track_ang_vel_z_exp": 5.0,
    "heading_error": -1.0,
    "dont_wait": -0.5,
    "is_alive": 3.0,
    "lin_vel_z_l2": -5.0,
    "stand_still": -1.0,
    # Robot-specific regularization
    "qingyun_hip_yaw_joint_sign_penalty": -10.0,
    "volume_points_penetration_feet": -1.0,
    "volume_points_penetration_knee": -1.0,
    "feet_slide": -1.0,
    "joint_deviation_upper_body": -0.01,
    "freeze_upper_torso": -0.8,
    "ang_vel_xy_l2": -0.1,
    "dof_torques_l2": -1.0e-5,
    "dof_acc_l2": -2.5e-7,
    "dof_vel_l2": -1.0e-4,
    "joint_regularization": -1.0e-4,
    "action_rate_l2": -0.01,
    "flat_orientation_l2": -3.0,
    "pelvis_orientation_l2": -3.0,
    "feet_flat_ori": -0.4,
    "feet_at_plane": -0.1,
    "terrain_adaptive_foot_lift": 3.0,
    "sound_suppression": -5.0e-4,
    "energy": -5.0e-5,
    # Safety rewards
    "dof_pos_limits": -1.0,
    "dof_vel_limits": -1.0,
    "torque_limits": -0.01,
    "undesired_contacts": -1.0,
    "feet_stumble": -1.0,
}

# Tune how strongly the swing foot should clear the terrain in front of the robot.
# Heights are in meters. The reward uses the torso height scanner to infer stairs, gaps, and slopes.
QINGYUN_TERRAIN_ADAPTIVE_FOOT_LIFT_PARAMS = {
    "height_scanner_cfg": SceneEntityCfg("height_scanner"),
    "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_pitch_link"),
    "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_pitch_link"),
    "command_name": "base_velocity",
    "forward_window": (0.15, 0.85),
    "support_window": (-0.15, 0.15),
    "lateral_window": 0.35,
    "base_clearance": 0.045,
    "stair_clearance_margin": 0.035,
    "down_stair_clearance": 0.06,
    "gap_clearance": 0.12,
    "gap_depth_scale": 0.5,
    "slope_clearance": 0.06,
    "terrain_threshold": 0.025,
    "gap_threshold": 0.10,
    "command_threshold": 0.10,
    "std": 0.055,
    "max_desired_clearance": 0.32,
    "use_command_direction": True,
    "use_terrain_type": True,
    "gap_terrain_keywords": ("gap",),
    "stair_terrain_keywords": ("pyramid_stairs",),
    "slope_terrain_keywords": ("slope",),
    "geometry_fallback": True,
}

# Tune QingYun curriculum schedules for reward weights here.
# These values override the inherited curriculum terms from parkour_env_cfg.py.
QINGYUN_VOLUME_POINTS_PENETRATION_WEIGHT_CURRICULUM = {
    "feet": {
        "term_name": "volume_points_penetration_feet",
        "init_weight": -1.0,
        "final_weight": -5.0,
        "lin_vel_threshold": (0.7, 0.9),
        "ang_vel_threshold": (0.0, 0.0),
        "step_size": 0.03,
    },
    "knee": {
        "term_name": "volume_points_penetration_knee",
        "init_weight": -1.0,
        "final_weight": -5.0,
        "lin_vel_threshold": (0.7, 0.9),
        "ang_vel_threshold": (0.0, 0.0),
        "step_size": 0.03,
    },
    "feet_stumble": {
        "term_name": "feet_stumble",
        "init_weight": -1.0,
        "final_weight": -5.0,
        "lin_vel_threshold": (0.7, 0.9),
        "ang_vel_threshold": (0.0, 0.0),
        "step_size": 0.03,
    },
}

QINGYUN_KEY_BODY_NAMES = [
    "lp_foot_pitch_link",
    "rp_foot_pitch_link",
    "lp_knee_pitch_link",
    "rp_knee_pitch_link",
    "lp_elbow_pitch_link",
    "rp_elbow_pitch_link",
]

QINGYUN_LINKS = [
    "base_link",
    "lp_hip_pitch_link",
    "rp_hip_pitch_link",
    "p_torso_yaw_link",
    "lp_hip_roll_link",
    "rp_hip_roll_link",
    "lp_shoulder_pitch_link",
    "rp_shoulder_pitch_link",
    "p_neck_yaw_link",
    "lp_hip_yaw_link",
    "rp_hip_yaw_link",
    "lp_arm_roll_link",
    "rp_arm_roll_link",
    "p_necks_pitch_link",
    "lp_knee_pitch_link",
    "rp_knee_pitch_link",
    "lp_arm_yaw_link",
    "rp_arm_yaw_link",
    "lp_foot_pitch_link",
    "rp_foot_pitch_link",
    "lp_elbow_pitch_link",
    "rp_elbow_pitch_link",
]

ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.values():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class QingYunRev30ParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = QINGYUN_Z1_A_REV_3_0_19_DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(QINGYUN_LINKS))

        self.scene.left_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/lp_foot_pitch_link"
        self.scene.right_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/rp_foot_pitch_link"
        self.scene.leg_volume_points.prim_path = "{ENV_REGEX_NS}/Robot/.*_foot_pitch_link"
        self.scene.knee_volume_points.prim_path = "{ENV_REGEX_NS}/Robot/.*_knee_pitch_link"
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/p_torso_yaw_link"
        self.scene.camera.offset.pos = (
            0.078000,
            0.000000,
            0.206000,
        )
        self.scene.camera.offset.rot = (
            0.866,
            0.0,
            0.5,
            0.0,
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/p_torso_yaw_link"

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "qingyun_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "36_01": 1,
            "36_11": 1,
            "114_08": 1,
            "114_09": 1,
            "A1-_Stand_stageii": 1,
            "B9_-__Walk_turn_left_90_stageii": 1,
            "B10_-__Walk_turn_left_45_stageii": 1,
            "B13_-__Walk_turn_right_90_stageii": 1,
            "B14_-__Walk_turn_right_45_t2_stageii": 1,
            "B15_-__Walk_turn_around_stageii": 1,
            "turn_l": 1,
            "turn_r": 1,

            # "move_back": 1,
            # "move_l": 1,
            # "move_r": 1,
        }

        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=QINGYUN_KEY_BODY_NAMES,
                preserve_order=True,
            )
        }

        self._apply_qingyun_volume_point_grids()
        self._apply_qingyun_velocity_commands()
        self._apply_qingyun_link_and_joint_names()
        self._apply_qingyun_reward_weights()
        self._apply_qingyun_curriculum_weights()

    def _apply_qingyun_volume_point_grids(self):
        self.scene.leg_volume_points.points_generator = copy.deepcopy(QINGYUN_LEG_VOLUME_POINTS_GRID)
        self.scene.knee_volume_points.points_generator = copy.deepcopy(QINGYUN_KNEE_VOLUME_POINTS_GRID)

    def _apply_qingyun_velocity_commands(self):
        command = self.commands.base_velocity
        command.resampling_time_range = QINGYUN_COMMAND_RESAMPLING_TIME_RANGE
        command.rel_standing_envs = QINGYUN_REL_STANDING_ENVS
        command.straight_target_prob = QINGYUN_STRAIGHT_TARGET_PROB
        command.ranges = mdp.PoseVelocityCommandCfg.Ranges(**QINGYUN_COMMAND_RANGES)
        command.velocity_ranges = copy.deepcopy(QINGYUN_TERRAIN_VELOCITY_RANGES)
        command.random_velocity_terrain = list(QINGYUN_RANDOM_VELOCITY_TERRAIN)
        command.only_positive_lin_vel_x = QINGYUN_ONLY_POSITIVE_LIN_VEL_X
        command.lin_vel_threshold = QINGYUN_LIN_VEL_THRESHOLD
        command.ang_vel_threshold = QINGYUN_ANG_VEL_THRESHOLD

    def _apply_qingyun_reward_weights(self):
        rewards = self.rewards.rewards
        for term_name, weight in QINGYUN_REWARD_WEIGHTS.items():
            term = getattr(rewards, term_name, None)
            if term is not None:
                term.weight = weight

    def _apply_qingyun_curriculum_weights(self):
        self.curriculum.volume_points_penetration_weight_feet.params = copy.deepcopy(
            QINGYUN_VOLUME_POINTS_PENETRATION_WEIGHT_CURRICULUM["feet"]
        )
        self.curriculum.volume_points_penetration_weight_knee.params = copy.deepcopy(
            QINGYUN_VOLUME_POINTS_PENETRATION_WEIGHT_CURRICULUM["knee"]
        )
        self.curriculum.feet_stumble_weight.params = copy.deepcopy(
            QINGYUN_VOLUME_POINTS_PENETRATION_WEIGHT_CURRICULUM["feet_stumble"]
        )

    def _apply_qingyun_link_and_joint_names(self):
        foot_body_pattern = ".*_foot_pitch_link"
        knee_body_pattern = ".*_knee_pitch_link"
        torso_body_name = "p_torso_yaw_link"

        self.rewards.rewards.feet_slide.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=foot_body_pattern
        )
        self.rewards.rewards.feet_slide.params["asset_cfg"] = SceneEntityCfg("robot", body_names=foot_body_pattern)
        self.rewards.rewards.joint_deviation_upper_body.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[".*_arm_.*_joint", ".*_elbow_.*_joint", "waist_yaw_joint"],
        )
        self.rewards.rewards.freeze_upper_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint"]
        )
        self.rewards.rewards.pelvis_orientation_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=torso_body_name
        )
        self.rewards.rewards.feet_flat_ori.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=foot_body_pattern
        )
        self.rewards.rewards.feet_flat_ori.params["asset_cfg"] = SceneEntityCfg("robot", body_names=foot_body_pattern)
        self.rewards.rewards.feet_at_plane.params["contact_sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=foot_body_pattern
        )
        self.rewards.rewards.feet_at_plane.params["asset_cfg"] = SceneEntityCfg("robot", body_names=foot_body_pattern)
        self.rewards.rewards.sound_suppression.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=foot_body_pattern
        )
        self.rewards.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=f"(?!{foot_body_pattern}).*"
        )
        self.rewards.rewards.feet_stumble.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[foot_body_pattern, knee_body_pattern]
        )
        self.rewards.rewards.terrain_adaptive_foot_lift = RewTerm(
            func=mdp.terrain_adaptive_foot_lift,
            weight=QINGYUN_REWARD_WEIGHTS["terrain_adaptive_foot_lift"],
            params=copy.deepcopy(QINGYUN_TERRAIN_ADAPTIVE_FOOT_LIFT_PARAMS),
        )
        self.rewards.rewards.rpo_thigh_yaw_joint_sign_penalty = None
        self.rewards.rewards.qingyun_hip_yaw_joint_sign_penalty = RewTerm(
            func=mdp.qingyun_hip_yaw_joint_sign_penalty,
            weight=-10.0,
        )

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=torso_body_name
        )
        self.terminations.root_height.params["minimum_height"] = QINGYUN_ROOT_HEIGHT_MIN

        self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[torso_body_name, "base_link"]
        )
        self.events.scale_link_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["lp_.*_link", "rp_.*_link"]
        )


@configclass
class QingYunRev30ParkourRoughEnvCfg_PLAY(QingYunRev30ParkourRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1

        self.scene.leg_volume_points.debug_vis = True
        self.scene.knee_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class QingYunRev30ParkourEnvCfg(QingYunRev30ParkourRoughEnvCfg):
    pass


@configclass
class QingYunRev30ParkourEnvCfg_PLAY(QingYunRev30ParkourRoughEnvCfg_PLAY):
    pass
