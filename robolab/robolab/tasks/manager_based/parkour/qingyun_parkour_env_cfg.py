import copy
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.qingyun_z1_A_rev_3_0 import QINGYUN_Z1_A_REV_3_0_19_DOF_CFG
from robolab.sensors import get_link_prim_targets
import robolab.tasks.manager_based.parkour.mdp as mdp
from robolab.tasks.manager_based.parkour.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg

AMP_NUM_STEPS = 3

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
            0.0875,
            0.0,
            0.20568,
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

        self._apply_qingyun_link_and_joint_names()

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
        self.rewards.rewards.rpo_thigh_yaw_joint_sign_penalty = None
        self.rewards.rewards.qingyun_hip_yaw_joint_sign_penalty = RewTerm(
            func=mdp.qingyun_hip_yaw_joint_sign_penalty,
            weight=-10.0,
        )

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=torso_body_name
        )

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
