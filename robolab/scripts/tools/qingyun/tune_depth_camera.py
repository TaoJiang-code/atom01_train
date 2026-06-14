from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tune the QingYun depth camera with an OpenCV GUI.")
parser.add_argument(
    "--robot",
    type=str,
    default="qingyun",
    choices=["qingyun", "atom01"],
    help="Robot configuration to preview. 'atom01' is preview-only and will not be patched.",
)
parser.add_argument(
    "--window_scale",
    type=int,
    default=10,
    help="Scale factor for the OpenCV depth preview window.",
)
parser.add_argument(
    "--raw_depth_clip",
    type=float,
    default=2.5,
    help="Maximum metric depth used for raw-depth visualization.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False):
    raise ValueError("This tool requires a graphical session. Please run it without --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.sim import SimulationContext

from robolab.tasks.manager_based.parkour.qingyun_parkour_env_cfg import QingYunRev30ParkourRoughEnvCfg
from robolab.tasks.manager_based.parkour.rpo_parkour_env_cfg import RPOParkourRoughEnvCfg


CONTROL_WINDOW = "QingYun Depth Camera Controls"
DEPTH_WINDOW = "QingYun First-Person Depth"
BUTTON_WINDOW = "Depth Camera Actions"
QINGYUN_CFG_PATH = (
    Path(__file__).resolve().parents[4]
    / "robolab"
    / "robolab"
    / "tasks"
    / "manager_based"
    / "parkour"
    / "qingyun_parkour_env_cfg.py"
)


@dataclass(frozen=True)
class SliderSpec:
    name: str
    min_value: float
    max_value: float
    scale: float

    @property
    def slider_max(self) -> int:
        return int(round((self.max_value - self.min_value) * self.scale))

    def to_slider(self, value: float) -> int:
        value = min(max(value, self.min_value), self.max_value)
        return int(round((value - self.min_value) * self.scale))

    def from_slider(self, raw_value: int) -> float:
        return self.min_value + raw_value / self.scale


SLIDER_SPECS = {
    "x": SliderSpec("x (m)", -0.30, 0.50, 1000.0),
    "y": SliderSpec("y (m)", -0.30, 0.30, 1000.0),
    "z": SliderSpec("z (m)", 0.00, 0.50, 1000.0),
    "roll": SliderSpec("roll (deg)", -180.0, 180.0, 10.0),
    "pitch": SliderSpec("pitch (deg)", -180.0, 180.0, 10.0),
    "yaw": SliderSpec("yaw (deg)", -180.0, 180.0, 10.0),
}

BUTTON_RECTS = {
    "save": ((20, 20), (180, 70)),
    "print": ((210, 20), (370, 70)),
    "reset": ((20, 90), (180, 140)),
}
PENDING_ACTION: dict[str, str | None] = {"value": None}
STATUS_TEXT = {"value": ""}


def _noop(_value: int) -> None:
    return None


def _button_mouse_callback(event: int, x: int, y: int, _flags: int, _userdata) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for action, (p0, p1) in BUTTON_RECTS.items():
        if p0[0] <= x <= p1[0] and p0[1] <= y <= p1[1]:
            PENDING_ACTION["value"] = action
            return


def build_scene() -> tuple[SimulationContext, InteractiveScene, Articulation]:
    env_cfg = QingYunRev30ParkourRoughEnvCfg() if args_cli.robot == "qingyun" else RPOParkourRoughEnvCfg()
    scene_cfg = env_cfg.scene
    scene_cfg.num_envs = 1
    scene_cfg.env_spacing = 2.5

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    robot: Articulation = scene["robot"]

    sim.reset()
    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        torch.zeros_like(robot.data.default_joint_vel),
    )
    scene.write_data_to_sim()
    sim.render()
    scene.update(sim.get_physics_dt())

    sim.set_camera_view([2.2, 2.2, 1.4], [0.0, 0.0, 0.8])
    return sim, scene, robot


def get_default_pose(scene: InteractiveScene) -> dict[str, float]:
    camera = scene.sensors["camera"]
    quat = torch.tensor([camera.cfg.offset.rot], dtype=torch.float32, device=camera.device)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    return {
        "x": float(camera.cfg.offset.pos[0]),
        "y": float(camera.cfg.offset.pos[1]),
        "z": float(camera.cfg.offset.pos[2]),
        "roll": math.degrees(float(roll[0])),
        "pitch": math.degrees(float(pitch[0])),
        "yaw": math.degrees(float(yaw[0])),
    }


def create_trackbars(default_pose: dict[str, float]) -> None:
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 620, 360)
    for key, spec in SLIDER_SPECS.items():
        cv2.createTrackbar(spec.name, CONTROL_WINDOW, spec.to_slider(default_pose[key]), spec.slider_max, _noop)


def set_trackbar_pose(pose: dict[str, float]) -> None:
    for key, spec in SLIDER_SPECS.items():
        cv2.setTrackbarPos(spec.name, CONTROL_WINDOW, spec.to_slider(pose[key]))


def read_trackbar_pose() -> dict[str, float]:
    pose: dict[str, float] = {}
    for key, spec in SLIDER_SPECS.items():
        pose[key] = spec.from_slider(cv2.getTrackbarPos(spec.name, CONTROL_WINDOW))
    return pose


def draw_action_window(enable_save: bool) -> None:
    canvas = np.full((170, 400, 3), 30, dtype=np.uint8)
    title = "QingYun actions" if enable_save else "Atom01 preview"
    cv2.putText(canvas, title, (18, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    active_buttons = ["print", "reset"] + (["save"] if enable_save else [])
    for action, (p0, p1) in BUTTON_RECTS.items():
        enabled = action in active_buttons
        color = (60, 140, 60) if enabled else (70, 70, 70)
        cv2.rectangle(canvas, p0, p1, color, thickness=-1)
        cv2.rectangle(canvas, p0, p1, (220, 220, 220), thickness=2)
        label = action.upper()
        cv2.putText(
            canvas,
            label,
            (p0[0] + 22, p0[1] + 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (245, 245, 245),
            2,
        )

    if STATUS_TEXT["value"]:
        cv2.putText(canvas, STATUS_TEXT["value"][:46], (18, 155 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 210, 120), 1)

    cv2.namedWindow(BUTTON_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(BUTTON_WINDOW, _button_mouse_callback)
    cv2.imshow(BUTTON_WINDOW, canvas)


def pose_to_quat_wxyz(pose: dict[str, float], device: str) -> torch.Tensor:
    roll = torch.tensor([math.radians(pose["roll"])], dtype=torch.float32, device=device)
    pitch = torch.tensor([math.radians(pose["pitch"])], dtype=torch.float32, device=device)
    yaw = torch.tensor([math.radians(pose["yaw"])], dtype=torch.float32, device=device)
    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    return math_utils.quat_unique(math_utils.normalize(quat))


def apply_camera_offset(scene: InteractiveScene, pose: dict[str, float]) -> torch.Tensor:
    camera = scene.sensors["camera"]
    env_ids = torch.tensor([0], device=camera.device, dtype=torch.long)

    offset_pos = torch.tensor(
        [[pose["x"], pose["y"], pose["z"]]],
        dtype=torch.float32,
        device=camera.device,
    )
    offset_quat = pose_to_quat_wxyz(pose, camera.device)
    camera_pos_w, camera_quat_w = obtain_world_pose_from_view(camera._view, env_ids)
    camera_pos_w, camera_quat_w = math_utils.combine_frame_transforms(
        camera_pos_w,
        camera_quat_w,
        offset_pos,
        offset_quat,
    )
    camera.set_world_poses(
        camera_pos_w,
        camera_quat_w,
        env_ids=env_ids,
        convention=camera.cfg.offset.convention,
    )
    return offset_quat[0]


def colorize_depth(depth: np.ndarray, clip_max: float) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=clip_max, posinf=clip_max, neginf=0.0)
    depth = np.clip(depth, 0.0, clip_max)
    if clip_max <= 0.0:
        clip_max = 1.0
    depth_u8 = np.round(depth / clip_max * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def colorize_normalized_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0)
    depth = np.clip(depth, 0.0, 1.0)
    depth_u8 = np.round(depth * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def render_depth_preview(scene: InteractiveScene, pose: dict[str, float], quat: torch.Tensor) -> None:
    camera = scene.sensors["camera"]
    raw_depth = camera.data.output["distance_to_image_plane"][0, :, :, 0].detach().cpu().numpy()
    noised_depth = camera.data.output["distance_to_image_plane_noised"][0, :, :, 0].detach().cpu().numpy()

    raw_bgr = colorize_depth(raw_depth, args_cli.raw_depth_clip)
    noised_bgr = colorize_normalized_depth(noised_depth)

    raw_bgr = cv2.resize(
        raw_bgr,
        (raw_bgr.shape[1] * args_cli.window_scale, raw_bgr.shape[0] * args_cli.window_scale),
        interpolation=cv2.INTER_NEAREST,
    )
    noised_bgr = cv2.resize(
        noised_bgr,
        (noised_bgr.shape[1] * args_cli.window_scale, noised_bgr.shape[0] * args_cli.window_scale),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.putText(raw_bgr, "raw metric depth", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(noised_bgr, "network input depth", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    canvas = np.hstack([raw_bgr, noised_bgr])
    overlay_lines = [
        f"pos=({pose['x']:.4f}, {pose['y']:.4f}, {pose['z']:.4f}) m",
        f"rpy=({pose['roll']:.2f}, {pose['pitch']:.2f}, {pose['yaw']:.2f}) deg",
        f"quat=({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})",
        "keys: q/ESC quit | r reset | p print config",
    ]
    y = canvas.shape[0] - 90
    for line in overlay_lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        y += 24

    cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)
    cv2.imshow(DEPTH_WINDOW, canvas)


def format_config_snippet(pose: dict[str, float], quat: torch.Tensor) -> str:
    return (
        "self.scene.camera.offset.pos = (\n"
        f"    {pose['x']:.6f},\n"
        f"    {pose['y']:.6f},\n"
        f"    {pose['z']:.6f},\n"
        ")\n"
        "self.scene.camera.offset.rot = (\n"
        f"    {float(quat[0]):.6f},\n"
        f"    {float(quat[1]):.6f},\n"
        f"    {float(quat[2]):.6f},\n"
        f"    {float(quat[3]):.6f},\n"
        ")"
    )


def save_qingyun_pose_to_cfg(pose: dict[str, float], quat: torch.Tensor) -> str:
    source = QINGYUN_CFG_PATH.read_text(encoding="utf-8")
    replacement = (
        "        self.scene.camera.offset.pos = (\n"
        f"            {pose['x']:.6f},\n"
        f"            {pose['y']:.6f},\n"
        f"            {pose['z']:.6f},\n"
        "        )\n"
        "        self.scene.camera.offset.rot = (\n"
        f"            {float(quat[0]):.6f},\n"
        f"            {float(quat[1]):.6f},\n"
        f"            {float(quat[2]):.6f},\n"
        f"            {float(quat[3]):.6f},\n"
        "        )"
    )
    pattern = re.compile(
        r"        self\.scene\.camera\.offset\.pos = \(\n"
        r"(?:\s+.*,\n){3}"
        r"        \)\n"
        r"        self\.scene\.camera\.offset\.rot = \(\n"
        r"(?:\s+.*,\n){4}"
        r"        \)",
        re.MULTILINE,
    )
    updated, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not locate the QingYun camera offset block in {QINGYUN_CFG_PATH}.")
    QINGYUN_CFG_PATH.write_text(updated, encoding="utf-8")
    return str(QINGYUN_CFG_PATH)


def main() -> None:
    sim, scene, _robot = build_scene()
    default_pose = get_default_pose(scene)
    tuning_enabled = args_cli.robot == "qingyun"
    if tuning_enabled:
        create_trackbars(default_pose)

    current_pose = dict(default_pose)
    current_quat = apply_camera_offset(scene, current_pose)

    if tuning_enabled:
        print("[INFO] QingYun depth camera tuner started.")
        print("[INFO] Use the OpenCV sliders to change x/y/z and roll/pitch/yaw.")
        print("[INFO] Click SAVE to write the tuned pose back into qingyun_parkour_env_cfg.py.")
    else:
        print("[INFO] Atom01 preview started in read-only mode.")
        print("[INFO] Slider-based tuning is intentionally disabled for atom01 preview.")
    print("[INFO] Press 'p' to print the current config snippet, 'r' to reset, and 'q' or ESC to quit.")
    print(format_config_snippet(current_pose, current_quat))

    try:
        while simulation_app.is_running():
            if tuning_enabled:
                new_pose = read_trackbar_pose()
                if any(abs(new_pose[key] - current_pose[key]) > 1.0e-6 for key in current_pose):
                    current_pose = new_pose
                    current_quat = apply_camera_offset(scene, current_pose)

            sim.render()
            scene.update(sim.get_physics_dt())
            render_depth_preview(scene, current_pose, current_quat)
            draw_action_window(enable_save=tuning_enabled)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                current_pose = dict(default_pose)
                if tuning_enabled:
                    set_trackbar_pose(current_pose)
                current_quat = apply_camera_offset(scene, current_pose)
                STATUS_TEXT["value"] = "Reset to default pose."
                print("[INFO] Camera pose reset to the current QingYun default.")
                print(format_config_snippet(current_pose, current_quat))
            if key == ord("p"):
                STATUS_TEXT["value"] = "Printed the current config snippet."
                print(format_config_snippet(current_pose, current_quat))

            action = PENDING_ACTION["value"]
            if action is not None:
                PENDING_ACTION["value"] = None
                if action == "print":
                    STATUS_TEXT["value"] = "Printed the current config snippet."
                    print(format_config_snippet(current_pose, current_quat))
                elif action == "reset":
                    current_pose = dict(default_pose)
                    if tuning_enabled:
                        set_trackbar_pose(current_pose)
                    current_quat = apply_camera_offset(scene, current_pose)
                    STATUS_TEXT["value"] = "Reset to default pose."
                    print("[INFO] Camera pose reset to the current default.")
                    print(format_config_snippet(current_pose, current_quat))
                elif action == "save":
                    if tuning_enabled:
                        saved_path = save_qingyun_pose_to_cfg(current_pose, current_quat)
                        STATUS_TEXT["value"] = "Saved pose back into QingYun cfg."
                        print(f"[INFO] Saved tuned pose to {saved_path}")
                        print(format_config_snippet(current_pose, current_quat))
                    else:
                        STATUS_TEXT["value"] = "Save is disabled for atom01 preview."
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    simulation_app.close()
'''
python robolab/scripts/tools/qingyun/tune_depth_camera.py --robot atom01

python robolab/scripts/tools/qingyun/tune_depth_camera.py --robot qingyun
'''