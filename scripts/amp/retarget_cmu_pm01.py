"""CMU ASF/AMC to PM01 compact NPZ using GMR and position-based limb targets."""
import argparse
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT


def read_asf(path):
    bones, parents, root = {}, {}, {}
    section, bone = None, None
    scale = None
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith('#'):
            continue
        key = words[0]
        if key.startswith(':'):
            section = key
        elif section == ':units' and key == 'length':
            scale = 0.0254 / float(words[1])
        elif section == ':root':
            root[key] = words[1:]
        elif section == ':bonedata':
            if key == 'begin':
                bone = {'dof': []}
            elif key == 'end':
                bones[bone['name'][0]] = bone
            elif key in ('name', 'direction', 'length', 'axis', 'dof'):
                bone[key] = words[1:]
        elif section == ':hierarchy' and key not in ('begin', 'end'):
            for child in words[1:]:
                parents[child] = key
    if scale is None or root.get('order') != ['TX', 'TY', 'TZ', 'RX', 'RY', 'RZ']:
        raise ValueError('Expected CMU ASF units and root channel order')
    if any(float(v) != 0 for v in root['orientation'] + root['position']):
        raise ValueError('Nonzero ASF root rest transform is not supported')
    return bones, parents, scale


def read_amc(path):
    text = Path(path).read_text()
    if ':DEGREES' not in text:
        raise ValueError('AMC rotations must be in degrees')
    frames = []
    for line in text.splitlines():
        words = line.split()
        if not words or words[0].startswith(('#', ':')):
            continue
        if words[0].isdigit():
            frames.append({})
        else:
            frames[-1][words[0]] = np.array(words[1:], dtype=float)
    return frames


def forward(frame, bones, parents, scale):
    # CMU is Y-up. Map forward +Z to robot +X and up +Y to +Z.
    world = R.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    positions = {'root': frame['root'][:3] * scale}
    rotations = {'root': R.from_euler('xyz', frame['root'][3:], degrees=True)}
    starts = {'root': positions['root']}

    def visit(name):
        if name in positions:
            return
        parent = parents[name]
        visit(parent)
        bone = bones[name]
        axis = R.from_euler(bone['axis'][3].lower(), np.array(bone['axis'][:3], float), degrees=True)
        channels = bone['dof']
        angles = dict(zip(channels, frame.get(name, [])))
        if len(angles) != len(channels):
            raise ValueError(f'Missing AMC channels for {name}')
        motion = R.from_euler('xyz', [angles.get(c, 0) for c in ('rx', 'ry', 'rz')], degrees=True)
        rotations[name] = rotations[parent] * axis * motion * axis.inv()
        starts[name] = positions[parent]
        offset = np.array(bone['direction'], float) * float(bone['length'][0]) * scale
        positions[name] = starts[name] + rotations[name].apply(offset)

    for name in bones:
        visit(name)
    return {name: [world.apply(starts[name]), (world * rotations[name] * world.inv()).as_quat(scalar_first=True)]
            for name in starts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asf', required=True)
    parser.add_argument('--amc', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float, required=True, help='Source AMC sampling rate; AMC does not store it')
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.fps <= 0:
        parser.error('--fps must be positive')
    bones, parents, scale = read_asf(args.asf)
    frames = read_amc(args.amc)
    if args.max_frames:
        frames = frames[:args.max_frames]
    mapping = {'LINK_BASE': 'root', 'LINK_TORSO_YAW': 'thorax',
               'LINK_HIP_ROLL_L': 'lfemur', 'LINK_HIP_ROLL_R': 'rfemur',
               'LINK_KNEE_PITCH_L': 'ltibia', 'LINK_KNEE_PITCH_R': 'rtibia',
               'LINK_ANKLE_ROLL_L': 'lfoot', 'LINK_ANKLE_ROLL_R': 'rfoot',
               'LINK_SHOULDER_ROLL_L': 'lhumerus', 'LINK_SHOULDER_ROLL_R': 'rhumerus',
               'LINK_ELBOW_PITCH_L': 'lradius', 'LINK_ELBOW_PITCH_R': 'rradius',
               'LINK_ELBOW_END_L': 'lwrist', 'LINK_ELBOW_END_R': 'rwrist'}
    # Scale by leg length rather than applying LAFAN-specific orientation offsets.
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT['engineai_pm01']))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    human_leg = sum(float(bones[n]['length'][0]) for n in ('lfemur', 'ltibia')) * scale
    robot_leg = sum(np.linalg.norm(data.body(a).xpos - data.body(b).xpos) for a, b in
                    [('LINK_HIP_ROLL_L', 'LINK_KNEE_PITCH_L'), ('LINK_KNEE_PITCH_L', 'LINK_ANKLE_ROLL_L')])
    table = {}
    for body, human in mapping.items():
        # Root orientation is anatomical; limb rotations differ across ASF/BVH,
        # so solve those using joint positions instead of BVH orientation offsets.
        table[body] = [human, 100 if 'ANKLE' in body else 10, 10 if human == 'root' else 0,
                       [0, 0, 0], data.body(body).xquat.tolist()]
    config = dict(robot_root_name='LINK_BASE', human_root_name='root', ground_height=0,
                  human_height_assumption=1.8, use_ik_match_table1=True, use_ik_match_table2=False,
                  human_scale_table={n: robot_leg / human_leg for n in mapping.values()},
                  ik_match_table1=table, ik_match_table2={})
    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / 'cmu.json'
        config_path.write_text(json.dumps(config))
        IK_CONFIG_DICT['cmu_amc_local'] = {'engineai_pm01': str(config_path)}
        retargeter = GeneralMotionRetargeting('cmu_amc_local', 'engineai_pm01', verbose=False)
        trajectory = np.stack([retargeter.retarget(forward(f, bones, parents, scale)).copy()
                               for f in tqdm(frames, desc='CMU → PM01')])
    if not np.isfinite(trajectory).all():
        raise ValueError('Nonfinite retargeted motion')
    addresses = []
    for number in range(24):
        matches = [j for j in range(model.njnt) if (model.joint(j).name or '').startswith(f'J{number:02d}_')]
        if len(matches) != 1:
            raise ValueError(f'Missing/ambiguous PM01 joint J{number:02d}')
        addresses.append(model.jnt_qposadr[matches[0]])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, fps=np.array(args.fps), root_pos=trajectory[:, :3],
                        root_rot=trajectory[:, [4, 5, 6, 3]], dof_pos=trajectory[:, addresses])
    print(f'Saved {output}: {len(trajectory)} frames. Inspect replay before training.')


if __name__ == '__main__':
    main()
