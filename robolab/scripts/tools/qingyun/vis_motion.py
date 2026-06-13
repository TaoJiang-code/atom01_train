"""Visualize a QingYun Rev3 Lab motion together with its mirrored counterpart."""

from __future__ import annotations

import argparse
import pickle

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay a QingYun Lab motion and its mirrored motion.")
parser.add_argument("--file", "-f", type=str, required=True, help="Path to the QingYun Lab .pkl motion file.")
parser.add_argument(
    "--show_key_bodies",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Visualize key-body markers for the original and mirrored motions.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robolab.assets.robots.qingyun_z1_A_rev_3_0 import QINGYUN_Z1_A_REV_3_0_19_DOF_CFG
from robolab.tasks.manager_based.parkour.mdp.symmetry import qingyun as qingyun_symmetry


def load_motion(path: str) -> dict:
    with open(path, "rb") as f:
        motion = pickle.load(f)

    required_keys = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = required_keys.difference(motion.keys())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Motion file is missing required keys: {missing_text}")

    dof_count = motion["dof_pos"].shape[1]
    if dof_count != qingyun_symmetry.QINGYUN_NUM_JOINTS:
        raise ValueError(
            "This viewer expects a QingYun Rev3 Lab motion with "
            f"{qingyun_symmetry.QINGYUN_NUM_JOINTS} DOFs, but the file contains {dof_count}. "
            "If this is an older 23-DOF dataset, regenerate it with the updated QingYun Rev3 retarget pipeline."
        )

    return motion


def mirror_motion_data(motion: dict, device: str) -> dict:
    root_pos = torch.as_tensor(motion["root_pos"], dtype=torch.float32, device=device).clone()
    root_rot = torch.as_tensor(motion["root_rot"], dtype=torch.float32, device=device).clone()
    dof_pos = torch.as_tensor(motion["dof_pos"], dtype=torch.float32, device=device).clone()

    root_pos[:, 1] *= -1.0

    rot_mat = math_utils.matrix_from_quat(root_rot)
    reflection = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=device, dtype=root_rot.dtype))
    mirrored_rot_mat = reflection.unsqueeze(0) @ rot_mat @ reflection.unsqueeze(0)
    mirrored_root_rot = math_utils.quat_from_matrix(mirrored_rot_mat)
    mirrored_root_rot = math_utils.quat_unique(math_utils.normalize(mirrored_root_rot))

    mirrored_dof = qingyun_symmetry._switch_joints_left_right(dof_pos)

    mirrored = {
        "fps": float(motion["fps"]),
        "root_pos": root_pos,
        "root_rot": mirrored_root_rot,
        "dof_pos": mirrored_dof,
        "loop_mode": motion.get("loop_mode", 0),
    }

    if "key_body_pos" in motion:
        key_body_pos = torch.as_tensor(motion["key_body_pos"], dtype=torch.float32, device=device).clone()
        key_body_pos[:, :, 1] *= -1.0
        mirrored["key_body_pos"] = key_body_pos

    return mirrored


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot = QINGYUN_Z1_A_REV_3_0_19_DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def make_key_body_markers() -> tuple[VisualizationMarkers, VisualizationMarkers]:
    original_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/QingYunMotionOriginal",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            )
        },
    )
    mirrored_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/QingYunMotionMirrored",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
            )
        },
    )
    return VisualizationMarkers(original_cfg), VisualizationMarkers(mirrored_cfg)


def run_simulator(sim: SimulationContext, scene: InteractiveScene, motions: list[dict]) -> None:
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    num_frames = min(motion["dof_pos"].shape[0] for motion in motions)
    env_origins = scene.env_origins[:, :3]

    key_body_markers = make_key_body_markers() if args_cli.show_key_bodies else None
    frame_idx = 0

    while simulation_app.is_running():
        root_states = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)

        for env_id, motion in enumerate(motions):
            root_states[env_id, :3] = motion["root_pos"][frame_idx] + env_origins[env_id]
            root_states[env_id, 3:7] = motion["root_rot"][frame_idx]
            joint_pos[env_id, :] = motion["dof_pos"][frame_idx]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        if key_body_markers is not None and all("key_body_pos" in motion for motion in motions):
            for env_id, marker in enumerate(key_body_markers):
                marker.visualize(
                    translations=motions[env_id]["key_body_pos"][frame_idx] + env_origins[env_id].unsqueeze(0)
                )

        midpoint = root_states[:, :3].mean(dim=0).cpu().numpy()
        sim.set_camera_view(midpoint + torch.tensor([2.5, 2.5, 1.3]).numpy(), midpoint)

        frame_idx = (frame_idx + 1) % num_frames


def main() -> None:
    motion = load_motion(args_cli.file)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(motion["fps"])
    sim = SimulationContext(sim_cfg)

    motion_tensors = {
        key: torch.as_tensor(value, dtype=torch.float32, device=sim.device) if hasattr(value, "shape") else value
        for key, value in motion.items()
    }
    mirrored_motion = mirror_motion_data(motion_tensors, sim.device)

    scene_cfg = ReplaySceneCfg(num_envs=2, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print(f"[INFO] Loaded motion: {args_cli.file}")
    print(f"[INFO] Frames: {motion_tensors['dof_pos'].shape[0]}, FPS: {float(motion_tensors['fps']):.1f}")
    print("[INFO] Environment 0 shows the original motion; environment 1 shows the mirrored motion.")

    run_simulator(sim, scene, [motion_tensors, mirrored_motion])


if __name__ == "__main__":
    main()
    simulation_app.close()

'''
python robolab/scripts/tools/qingyun/vis_motion.py \
  --file robolab/data/motions/qingyun_lab/36_01.pkl
'''