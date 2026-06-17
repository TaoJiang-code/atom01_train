import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from robolab.assets import ISAAC_ASSET_DIR


# Isaac Lab joint names for qingyun_z1_A_rev_3_0:
# - lw_hip_pitch_joint
# - rw_hip_pitch_joint
# - waist_yaw_joint
# - lw_hip_roll_joint
# - rw_hip_roll_joint
# - lw_shoulder_pitch_joint
# - rw_shoulder_pitch_joint
# - lw_hip_yaw_joint
# - rw_hip_yaw_joint
# - lw_arm_roll_joint
# - rw_arm_roll_joint
# - lw_knee_pitch_joint
# - rw_knee_pitch_joint
# - lw_arm_yaw_joint
# - rw_arm_yaw_joint
# - lw_foot_pitch_joint
# - rw_foot_pitch_joint
# - lw_elbow_pitch_joint
# - rw_elbow_pitch_joint
# Body: base_link, Mass: 1.7999999523162842 kg
# Body: lp_hip_pitch_link, Mass: 0.47999995946884155 kg
# Body: rp_hip_pitch_link, Mass: 0.47999995946884155 kg
# Body: p_torso_yaw_link, Mass: 5.5 kg
# Body: lp_hip_roll_link, Mass: 0.16870999336242676 kg
# Body: rp_hip_roll_link, Mass: 0.16870999336242676 kg
# Body: lp_shoulder_pitch_link, Mass: 0.4099999964237213 kg
# Body: rp_shoulder_pitch_link, Mass: 0.4099999964237213 kg
# Body: p_neck_yaw_link, Mass: 0.30000001192092896 kg
# Body: lp_hip_yaw_link, Mass: 0.4710000157356262 kg
# Body: rp_hip_yaw_link, Mass: 0.4710000157356262 kg
# Body: lp_arm_roll_link, Mass: 0.06822100281715393 kg
# Body: rp_arm_roll_link, Mass: 0.06822100281715393 kg
# Body: p_necks_pitch_link, Mass: 0.1850000023841858 kg
# Body: lp_knee_pitch_link, Mass: 1.159999966621399 kg
# Body: rp_knee_pitch_link, Mass: 1.159999966621399 kg
# Body: lp_arm_yaw_link, Mass: 0.5809999704360962 kg
# Body: rp_arm_yaw_link, Mass: 0.5809999704360962 kg
# Body: lp_foot_pitch_link, Mass: 0.5199999809265137 kg
# Body: rp_foot_pitch_link, Mass: 0.5199999809265137 kg
# Body: lp_elbow_pitch_link, Mass: 0.07928899675607681 kg
# Body: rp_elbow_pitch_link, Mass: 0.07928899675607681 kg

QINGYUN_Z1_A_REV_3_0_19_DOF_LAB_DOF_NAMES = [
    "lw_hip_pitch_joint",
    "rw_hip_pitch_joint",
    "waist_yaw_joint",
    "lw_hip_roll_joint",
    "rw_hip_roll_joint",
    "lw_shoulder_pitch_joint",
    "rw_shoulder_pitch_joint",
    "lw_hip_yaw_joint",
    "rw_hip_yaw_joint",
    "lw_arm_roll_joint",
    "rw_arm_roll_joint",
    "lw_knee_pitch_joint",
    "rw_knee_pitch_joint",
    "lw_arm_yaw_joint",
    "rw_arm_yaw_joint",
    "lw_foot_pitch_joint",
    "rw_foot_pitch_joint",
    "lw_elbow_pitch_joint",
    "rw_elbow_pitch_joint",
]

QINGYUN_Z1_A_REV_3_0_19_DOF_MJCF_DOF_NAMES = [
    "rw_hip_pitch_joint_motor",
    "rw_hip_roll_joint_motor",
    "rw_hip_yaw_joint_motor",
    "rw_knee_pitch_joint_motor",
    "rw_foot_pitch_joint_motor",
    "lw_hip_pitch_joint_motor",
    "lw_hip_roll_joint_motor",
    "lw_hip_yaw_joint_motor",
    "lw_knee_pitch_joint_motor",
    "lw_foot_pitch_joint_motor",
    "waist_yaw_joint_motor",
    "rw_shoulder_pitch_joint_motor",
    "rw_arm_roll_joint_motor",
    "rw_arm_yaw_joint_motor",
    "rw_elbow_pitch_joint_motor",
    "lw_shoulder_pitch_joint_motor",
    "lw_arm_roll_joint_motor",
    "lw_arm_yaw_joint_motor",
    "lw_elbow_pitch_joint_motor",
]

@configclass
class qingyun_z1_A_rev_3_0_19_dof_ArticulationCfg(ArticulationCfg):
    """Configuration for the qingyun_z1_A_rev_3_0 19 dof articulation."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


QINGYUN_Z1_A_REV_3_0_19_DOF_CFG = qingyun_z1_A_rev_3_0_19_dof_ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            f"{ISAAC_ASSET_DIR}/QingYun_Robot_Description/qingyun_z1_A_rev_3_0_description/usd/"
            "qingyun_z1_A_rev_3_0.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={
            "lw_hip_pitch_joint": 0.0,
            "rw_hip_pitch_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "lw_hip_roll_joint": 0.0,
            "rw_hip_roll_joint": 0.0,
            "lw_shoulder_pitch_joint": 0.0,
            "rw_shoulder_pitch_joint": 0.0,
            "lw_hip_yaw_joint": 0.0,
            "rw_hip_yaw_joint": 0.0,
            "lw_arm_roll_joint": 0.0,
            "rw_arm_roll_joint": 0.0,
            "lw_knee_pitch_joint": 0.0,
            "rw_knee_pitch_joint": 0.0,
            "lw_arm_yaw_joint": 0.0,
            "rw_arm_yaw_joint": 0.0,
            "lw_foot_pitch_joint": 0.0,
            "rw_foot_pitch_joint": 0.0,
            "lw_elbow_pitch_joint": 0.0,
            "rw_elbow_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 7.0,
                ".*_arm_roll_joint": 7.0,
                ".*_arm_yaw_joint": 4.0,
                ".*_elbow_pitch_joint": 4.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 41.89,
                ".*_arm_roll_joint": 41.89,
                ".*_arm_yaw_joint": 12.6,
                ".*_elbow_pitch_joint": 12.6,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_arm_roll_joint": 10.0,
                ".*_arm_yaw_joint": 10.0,
                ".*_elbow_pitch_joint": 10.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 0.7,
                ".*_arm_roll_joint": 0.7,
                ".*_arm_yaw_joint": 0.7,
                ".*_elbow_pitch_joint": 0.7,
            },
            armature=0.01,
            friction=0.0001,
            min_delay=5,
            max_delay=7,
        ),

        "waist": DelayedPDActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                # "waist_yaw_joint": 17.0,
                "waist_yaw_joint": 10.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 28.8,
            },
            stiffness={
                "waist_yaw_joint": 15.0,
            },
            damping={
                "waist_yaw_joint": 1.0,
            },
            armature=0.01,
            friction=0.0001,
            min_delay=5,
            max_delay=7,
        ),

        "leg": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_pitch_joint",
                ".*_foot_pitch_joint",
            ],
            # effort_limit_sim={
            #     ".*_hip_pitch_joint": 27.0,
            #     ".*_hip_roll_joint": 27.0,
            #     ".*_hip_yaw_joint": 27.0,
            #     ".*_knee_pitch_joint": 27.0,
            #     ".*_foot_pitch_joint": 27.0
            # },
            # velocity_limit_sim={
            #     ".*_hip_pitch_joint": 5.45,
            #     ".*_hip_roll_joint": 5.45,
            #     ".*_hip_yaw_joint": 5.45,
            #     ".*_knee_pitch_joint": 5.45,
            #     ".*_foot_pitch_joint": 5.45
            # },
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            # stiffness={
            #     ".*_hip_pitch_joint": 42.5,
            #     ".*_hip_roll_joint": 30.0,
            #     ".*_hip_yaw_joint": 30.0,
            #     ".*_knee_pitch_joint": 30.0,
            #     ".*_foot_pitch_joint": 30.0,
            # },
            stiffness=100.0,
            damping={
                ".*_hip_pitch_joint": 4.29,
                ".*_hip_roll_joint": 3.0,
                ".*_hip_yaw_joint": 3.0,
                ".*_knee_pitch_joint": 3.0,
                ".*_foot_pitch_joint": 3.0,
            },
            armature=0.01,
            friction=0.0001,
            min_delay=5,
            max_delay=7,
        ),
    },

    joint_sdk_names=QINGYUN_Z1_A_REV_3_0_19_DOF_MJCF_DOF_NAMES,
)
