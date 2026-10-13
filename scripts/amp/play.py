"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ensure repository root is on Python path for Hydra registry imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# The editable install of engineai_rl_lab may resolve to a different checkout. setuptools
# appends its finder after PathFinder, so putting this workspace first makes it win.
sys.path.insert(0, str(REPO_ROOT / "source" / "engineai_rl_lab"))

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--video", action="store_true", help="Record a video and exit.")
parser.add_argument("--video_length", type=int, default=1000, help="Video length in simulation control steps.")
parser.add_argument("--skip_export", action="store_true", help="Skip ONNX and MNN export.")
parser.add_argument("--record_npz", type=str, help="Save environment 0 poses for MuJoCo replay, without cameras.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video_length <= 0:
    parser.error("--video_length must be positive")
if args_cli.video:
    args_cli.enable_cameras = True
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import subprocess
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import engineai_rl_lab.tasks  # noqa: F401



@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)


    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "play")
        print(f"[INFO] Recording video to: {video_folder}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix=Path(resume_path).stem,
            disable_logger=True,
        )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Inference only needs the actor. This also keeps older checkpoints playable
    # when the privileged critic observation layout changes.
    ppo_runner.load(
        resume_path,
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
    )

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    if not args_cli.skip_export:
        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        file_basename = os.path.basename(resume_path).split(".")[0]
        onnx_filename = file_basename + ".onnx"
        ppo_runner.export_policy_to_onnx(path=export_model_dir, filename=onnx_filename)

        onnx_file = os.path.join(export_model_dir, onnx_filename)
        mnn_file = os.path.join(export_model_dir, file_basename + ".mnn")
        if os.path.exists(onnx_file):
            if importlib.util.find_spec("MNN") is None:
                print("[WARN] MNN is not installed in the current Python environment. Skipping MNN conversion.")
            else:
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "MNN.tools.mnnconvert",
                            "-f",
                            "ONNX",
                            "--modelFile",
                            onnx_file,
                            "--MNNModel",
                            mnn_file,
                            "--bizCode",
                            "MNN",
                        ],
                        check=True,
                    )
                    print(f"Successfully converted to MNN: {mnn_file}")
                except subprocess.CalledProcessError as err:
                    print(f"[WARN] Failed to convert ONNX to MNN: {err}")
        else:
            print(f"ONNX file not found: {onnx_file}")


    # reset environment
    obs = env.get_observations()
    timestep = 0
    poses = []
    if args_cli.record_npz:
        robot = env.unwrapped.scene["robot"]
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.record_npz:
                poses.append((
                    robot.data.root_pos_w[0].cpu().numpy().copy(),
                    robot.data.root_quat_w[0].cpu().numpy()[[1, 2, 3, 0]].copy(),
                    robot.data.joint_pos[0].cpu().numpy().copy(),
                ))
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        timestep += 1
        if (args_cli.video or args_cli.record_npz) and timestep >= args_cli.video_length:
            break

    if args_cli.record_npz and poses:
        output = Path(args_cli.record_npz).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output, fps=1.0 / env.unwrapped.step_dt,
            root_pos=np.stack([p[0] for p in poses]),
            root_rot=np.stack([p[1] for p in poses]),
            dof_pos=np.stack([p[2] for p in poses]),
            joint_names=np.asarray(robot.joint_names),
        )
        print(f"[INFO] Saved {len(poses)} policy frames to {output}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
