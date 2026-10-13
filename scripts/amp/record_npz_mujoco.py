"""Record a compact AMP motion with named joints to MP4 without Isaac Sim."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--video", default="videos/pm01_run.mp4")
    parser.add_argument("--model", default=str(Path.home() / "engineai_robotics_native_sdk/assets/resource/pm01_edu.xml"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--view", choices=("default", "front"), default="default",
                        help="front: keep the camera ahead of the robot as it turns")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    with np.load(args.input_file, allow_pickle=False) as clip:
        fps = float(clip["fps"].item())
        positions = clip["root_pos"]
        rotations = clip["root_rot"]
        joints = clip["dof_pos"]
        names = clip["joint_names"].tolist() if "joint_names" in clip.files else None
    model = mujoco.MjModel.from_xml_path(str(Path(args.model).resolve()))
    if names is None:
        # Legacy PM01 compact files store J00..J23 in numeric SDK order.
        by_number = {}
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name and name.startswith("J") and name[1:3].isdigit():
                by_number[int(name[1:3])] = name
        if joints.shape[1] != 24 or set(by_number) != set(range(24)):
            raise ValueError("Unnamed motions require PM01 SDK order: 24 columns J00..J23")
        names = [by_number[i] for i in range(24)]
    data = mujoco.MjData(model)
    addresses = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Model is missing joint: {name}")
        addresses.append(model.jnt_qposadr[joint_id])
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise ValueError("Expected one floating robot base")
    root = model.jnt_qposadr[free_joints[0]]
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 135
    camera.elevation = -15
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    output = Path(args.video).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(np.ceil(len(joints) / fps * args.fps))
    print(f"Recording {frame_count} frames to {output}", flush=True)
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        with imageio.get_writer(str(output), fps=args.fps, codec="libx264") as writer:
            for frame in range(frame_count):
                index = min(int(frame * fps / args.fps), len(joints) - 1)
                data.qpos[:] = model.qpos0
                data.qpos[root:root + 3] = positions[index]
                data.qpos[root + 3:root + 7] = rotations[index][[3, 0, 1, 2]]
                data.qpos[addresses] = joints[index]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = positions[index]
                if args.view == "front":
                    x, y, z, w = rotations[index]
                    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    # MuJoCo azimuth describes the viewing direction toward the target.
                    camera.azimuth = np.degrees(yaw) + 180.0
                renderer.update_scene(data, camera=camera, scene_option=options)
                writer.append_data(renderer.render())
                if frame % args.fps == 0 or frame == frame_count - 1:
                    print(f"\rProgress: {frame + 1}/{frame_count} ({100 * (frame + 1) / frame_count:.0f}%)", end="", flush=True)
    print()
    print(f"Saved {output} ({frame_count} frames, {args.fps} FPS)")


if __name__ == "__main__":
    main()
