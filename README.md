# engineai_bhr_lab

基于 [EngineAI Robotics / engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab) 的 PM01 跑步 AMP 实验项目。主要围绕直线跑步、左右转弯、低速与停止行为进行训练，并提供参考动作处理、策略回放、MuJoCo 视频录制和 SDK sim2sim 流程。

原项目的 T800、whole body tracking 代码仍保留。本 README 以 PM01 跑步为主；[英文文档](README_EN.md) 目前保留上游说明，尚未同步本分支的跑步流程。`docs/` 中原有 GIF 为上游演示，不代表本分支模型的验证结果。

## 跑步仿真效果

以下为本项目的跑步仿真效果，非真机演示。

| 仿真效果一 | 仿真效果二 |
|:---:|:---:|
| <img src="docs/ezgif.com-video-to-gif-converter.gif" alt="PM01 跑步仿真效果一" width="480"/> | <img src="docs/ezgif.com-video-to-gif-converter(1).gif" alt="PM01 跑步仿真效果二" width="480"/> |

## 当前实现

- 任务：`AMP-Flat-PM01-v1`，IsaacLab 训练，PPO + AMP。
- 跑步参考：直跑与左右转弯各 10 秒，转弯版本的目标偏航角速度为 ±1.2 rad/s。
- 辅助参考：原慢走动作的前 4 秒、后 4 秒；这些片段不等同于纯站立动作。
- 脚间距奖励：跑动指令下惩罚左右脚踝横向间距小于 0.2 m，不设上限；测量的是节点间距，不是鞋边净空。
- 支持无相机策略运行并保存 NPZ，再用 MuJoCo 录制，避免依赖 Isaac Sim RTX 视频渲染。

当前默认速度指令为前进 1～3 m/s、转向 ±1.2 rad/s，随机站立比例为 0.1；AMP 默认 `style_reward_scale=7.5`、`task_style_lerp=0.35`。命令行可覆盖这些值，以每次训练保存的 `params/env.yaml`、`params/agent.yaml` 为准。

目前无条件 AMP 模式下，零指令仍会获得风格奖励；停止能力需要单独验证。当前 checkpoint 保存逻辑未包含 AMP 判别器状态，续训时判别器会重新初始化。本文不将这两项描述为已修复功能。

## 安装与资产

上游基于 Isaac Sim 5.1 / IsaacLab 2.3.2 开发，IsaacLab 基准提交为 `c22775241e28f465fe345fa1a482ad6d29d712b0`。请先准备匹配的 IsaacLab 环境。

```bash
git clone https://github.com/zsp1018/engineai_bhr_lab.git
cd engineai_bhr_lab
git lfs install
git lfs pull

# 在已配置 IsaacLab 的 Python 环境中
python -m pip install -e source/engineai_rl_lab
python -m pip install mujoco imageio imageio-ffmpeg tensorboard
```

`datasets/amp/pm01/data` 是训练数据目录，纳入版本管理；克隆后必须获取 LFS 实体文件。运行还需要完整的 PM01 USD 资产，检查 `source/engineai_rl_lab/engineai_rl_lab/assets/pm01/`。如果该目录未随仓库提供，需从上游补齐，包括其 configuration 引用文件。

本机使用 `env_isaaclab`。`train_local.sh` / `play_local.sh` 是本机依赖启动脚本，要求以下本地目录存在：

- `.deps/rsl_rl_5`：rsl-rl-lib 5.0.0。
- `.deps/warp_1_8_1`：warp-lang 1.8.1。
- `.deps/cuda_driver`：用于本机 CUDA 驱动动态库查找。

`.deps` 不应作为跨机器可用环境直接分发。新机器需先配置依赖；已有兼容环境可直接运行对应的 `train.py` / `play.py`。SDK 和驱动路径同样需要按机器调整。

## 跑步参考数据

训练显式指定 [dataset_run_three.yaml](datasets/amp/pm01/config/dataset_run_three.yaml)。文件名沿用早期命名，目前包含 **5 段**动作：

| NPZ 文件 | 时长 | 权重 |
|---|---:|---:|
| `run2_subject1_straight_loop_10s_symmetric.npz` | 10 s | 1.0 |
| `run2_subject1_left_turn_loop_10s_symmetric_1p2.npz` | 10 s | 1.0 |
| `run2_subject1_right_turn_loop_10s_symmetric_1p2.npz` | 10 s | 1.0 |
| `walk_straight_slow_001_Skeleton9_first_4s.npz` | 4 s | 0.5 |
| `walk_straight_slow_001_Skeleton9_last_4s.npz` | 4 s | 0.5 |

无条件采样时按 YAML 权重选择片段，不按时长分配：三段跑步各约 25%，两段慢走各约 12.5%。修改数据配置后需重新启动训练才能加载。

紧凑 NPZ 包含 `fps`、`root_pos`、`root_rot`（xyzw）和 `dof_pos`。上述未命名 PM01 文件的关节列为 SDK J00～J23 顺序，加载器通过 YAML 的 `dof_indices` 对齐训练关节，并按 `zero_dof_indices` 处理腰部。不要随意删除映射。

左右转轨迹是路径和根节点朝向的运动学增强，保持原关节动作，尚未验证对应急转动作的动态平衡；播放正常不等于策略可执行。

## 训练与续训

以下命令在项目根目录、已准备好本机依赖的环境中运行：

```bash
conda activate env_isaaclab

bash scripts/amp/train_local.sh \
  --task AMP-Flat-PM01-v1 \
  --headless \
  --num_envs 4096 \
  --run_name pm01_running \
  --max_iterations 20000 \
  --logger tensorboard \
  agent.amp.dataset_path=datasets/amp/pm01/config/dataset_run_three.yaml \
  'env.commands.base_velocity.ranges.lin_vel_x=[1.0,3.0]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]' \
  'env.commands.base_velocity.ranges.ang_vel_z=[-1.2,1.2]' \
  env.commands.base_velocity.rel_standing_envs=0.1 \
  env.commands.base_velocity.heading_command=False
```

续训时，在上面命令中增加以下参数，替换为自己的训练目录和模型名：

```bash
--resume \
--load_run RUN_DIRECTORY \
--checkpoint model_20000.pt
```

`--max_iterations` 表示本次额外训练轮数。模型加载不会自动恢复旧 YAML 的全部超参数；需要保持的参数应显式传入。新日志位于 `logs/rsl_rl/amp_velocity_flat_pm01/` 下的新时间戳目录。GPU 资源不足时可降低 `--num_envs`。

```bash
# 查看训练曲线，浏览器打开 http://localhost:6006
tensorboard --logdir logs/rsl_rl/amp_velocity_flat_pm01 --port 6006
```

重点观察速度跟踪误差、`Step_Reward/style_reward`、判别器得分、终止比例，并结合视频判断。风格奖励数值不是动作相似度百分比。

## 策略回放与视频录制

替换 `RUN_DIRECTORY` 和 checkpoint。下面固定前进 2.8 m/s、左转 0.4 rad/s，运行 1000 个控制步（当前配置约 10 秒）：

```bash
bash scripts/amp/play_local.sh \
  --task AMP-Flat-PM01-v1 \
  --num_envs 1 --headless --skip_export \
  --load_run RUN_DIRECTORY \
  --checkpoint model_20000.pt \
  --record_npz videos/pm01_policy.npz \
  --video_length 1000 \
  agent.amp.dataset_path=datasets/amp/pm01/config/dataset_run_three.yaml \
  'env.commands.base_velocity.ranges.lin_vel_x=[2.8,2.8]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]' \
  'env.commands.base_velocity.ranges.ang_vel_z=[0.4,0.4]' \
  env.commands.base_velocity.rel_standing_envs=0.0 \
  env.commands.base_velocity.heading_command=False
```

`--record_npz` 保存模型在 IsaacLab 中实际产生的轨迹。转速正值左转，负值右转，零值直跑；目标速度不等于实际速度。测试停止时将三个速度范围全部设为 `[0.0,0.0]`。

```bash
python scripts/amp/record_npz_mujoco.py \
  --input_file videos/pm01_policy.npz \
  --video videos/pm01_policy_front.mp4 \
  --view front --fps 30
```

录制器默认使用 `~/engineai_robotics_native_sdk/assets/resource/pm01_edu.xml`，可通过 `--model` 指定其他路径。需要可用的 MuJoCo EGL 渲染环境。本机若已准备 `.deps/egl/10_nvidia.json`，可在录制命令前添加 `__EGL_VENDOR_LIBRARY_FILENAMES="$PWD/.deps/egl/10_nvidia.json"`。

`--view front` 为正前方朝向跟随视角，容易掩盖转弯；`--view default` 固定镜头方位并跟随位置。查看参考动作时，把 `--input_file` 换成数据集里的 NPZ 即可。

**这个流程是轨迹渲染，不是策略在 MuJoCo 中闭环运行的 sim2sim。**

## 导出与 MuJoCo sim2sim

1. 运行上述 `play_local.sh` 时去掉 `--skip_export`，可将 `--video_length` 改为 `1`。脚本导出 `exported/model_XXXXX.onnx`；安装 MNN 后还会尝试转换同名 `.mnn`，务必确认转换成功并实际生成文件。
2. 安装 [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk)，将 MNN 放入 `assets/config/pm01_edu/rl_lab/policies/`，修改 `rl_lab/default.yaml` 的 `policy_file` 指向该文件。替换已有策略前先备份。
3. 核对 SDK 的观测顺序、历史长度、关节映射、动作缩放、默认角度、PD 参数及控制周期与训练一致。AMP 跑步策略不需要配置 tracking 的参考轨迹文件。
4. 分别在 SDK 容器中的三个终端运行：

```bash
# 终端一：MuJoCo
./scripts/run_mujoco.sh pm01_edu

# 终端二：控制器，明确使用仿真平台
export ENGINEAI_ROBOTICS_PLATFORM=mujoco
./run.sh pm01_edu

# 终端三：虚拟手柄
python3 tools/virtual_gamepad/virtual_gamepad.py
```

每个终端先执行 `engineai_robotics_env` 进入容器，并切换到 SDK 根目录。按 SDK 当前状态机进入阻尼、PD 站立，再进入 `rl_lab`；本机 PM01 配置依次为 `LB+RB`、`LB+A`、`LB+X`。键位以所安装 SDK 的配置为准。

本分支主要用于跑步训练与仿真验证，未将当前跑步模型声明为已完成真机验证。真机部署请依据 SDK 文档和实际测试结果进行。

## 许可证
本项目采用 **BSD 3-Clause License** 开源协议。详见 [LICENSE](LICENSE.txt) 文件。

## 致谢

本项目得益于以下开源项目的支持和贡献，在此表示衷心的感谢：

- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** — 训练和运行仿真实验的基础框架。
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** — 适用于足式机器人的高性能强化学习库。
- **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** — 项目结构启发及有价值的功能实现参考。
- **[MNN](https://github.com/alibaba/mnn)** — 轻量级高性能推理引擎，用于端侧部署。
- **[Legged Lab](https://github.com/zitongbai/legged_lab)** — 基于 Isaac Lab 的足式机器人强化学习扩展，为本项目的 AMP 任务结构与实现提供了参考。
- **[MimicKit](https://github.com/xbpeng/MimicKit)** — 轻量级动作模仿方法库，为 AMP 判别器训练、经验回放和参考动作采样提供了实现参考。
- **[AMP for Hardware](https://github.com/escontra/AMP_for_hardware)** — 论文 *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions* 的代码，为 AMP 在真实机器人上的应用提供了参考。
