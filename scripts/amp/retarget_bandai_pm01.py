"""Retarget Bandai Namco six-channel BVH to PM01 using positional GMR targets."""
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
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bvh', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    text = Path(args.bvh).read_text()
    fps = 1 / float(next(line.split(':')[1] for line in text.splitlines() if line.startswith('Frame Time:')))
    channels = [line.split()[1:] for line in text.splitlines() if line.strip().startswith('CHANNELS')]
    if any(c != ['6', 'Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation'] for c in channels):
        raise ValueError('Expected Bandai six-channel ZXY BVH')
    motion = read_bvh(args.bvh)
    _, positions = quat_fk(motion.quats, motion.pos, motion.parents)
    # Bandai exporter stores local translations including bone offsets in channels.
    world = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    positions = positions @ world.T / 100
    index = {name: i for i, name in enumerate(motion.bones)}
    mapping = {'LINK_BASE': 'Hips', 'LINK_TORSO_YAW': 'Chest',
               'LINK_HIP_ROLL_L': 'UpperLeg_L', 'LINK_HIP_ROLL_R': 'UpperLeg_R',
               'LINK_KNEE_PITCH_L': 'LowerLeg_L', 'LINK_KNEE_PITCH_R': 'LowerLeg_R',
               'LINK_ANKLE_ROLL_L': 'Foot_L', 'LINK_ANKLE_ROLL_R': 'Foot_R',
               'LINK_SHOULDER_ROLL_L': 'UpperArm_L', 'LINK_SHOULDER_ROLL_R': 'UpperArm_R',
               'LINK_ELBOW_PITCH_L': 'LowerArm_L', 'LINK_ELBOW_PITCH_R': 'LowerArm_R',
               'LINK_ELBOW_END_L': 'Hand_L', 'LINK_ELBOW_END_R': 'Hand_R'}
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT['engineai_pm01']))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    human_leg = sum(np.linalg.norm(positions[0,index[a]] - positions[0,index[b]]) for a,b in
                    [('UpperLeg_L','LowerLeg_L'),('LowerLeg_L','Foot_L')])
    robot_leg = sum(np.linalg.norm(data.body(a).xpos-data.body(b).xpos) for a,b in
                    [('LINK_HIP_ROLL_L','LINK_KNEE_PITCH_L'),('LINK_KNEE_PITCH_L','LINK_ANKLE_ROLL_L')])
    table = {body: [human, 100 if 'ANKLE' in body else 10, 10 if human == 'Hips' else 0,
                    [0,0,0], data.body(body).xquat.tolist()] for body,human in mapping.items()}
    config = dict(robot_root_name='LINK_BASE',human_root_name='Hips',ground_height=0,
                  human_height_assumption=1.8,use_ik_match_table1=True,use_ik_match_table2=False,
                  human_scale_table={name:robot_leg/human_leg for name in mapping.values()},
                  ik_match_table1=table,ik_match_table2={})
    frames = []
    for p in positions:
        up = p[index['Chest']]-p[index['Hips']]
        up /= np.linalg.norm(up)
        left = p[index['UpperLeg_L']]-p[index['UpperLeg_R']]
        left -= up*np.dot(up,left)
        left /= np.linalg.norm(left)
        forward = np.cross(left,up)
        quat = R.from_matrix(np.column_stack([forward,left,up])).as_quat(scalar_first=True)
        frames.append({name:[p[index[name]].copy(),quat.copy()] for name in mapping.values()})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'bandai.json'
        path.write_text(json.dumps(config))
        IK_CONFIG_DICT['bandai_local']={'engineai_pm01':str(path)}
        retargeter = GeneralMotionRetargeting('bandai_local','engineai_pm01',verbose=False)
        q = np.stack([retargeter.retarget(frame).copy() for frame in tqdm(frames,desc='Bandai → PM01')])
    addresses=[]
    for number in range(24):
        ids=[j for j in range(model.njnt) if (model.joint(j).name or '').startswith(f'J{number:02d}_')]
        if len(ids)!=1:
            raise ValueError(f'Missing PM01 joint {number}')
        addresses.append(model.jnt_qposadr[ids[0]])
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite retargeted motion')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,fps=np.array(fps),root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,addresses])
    print(f'Saved {output}: {len(q)} frames, {fps:g} FPS. Inspect replay before training.')


if __name__ == '__main__':
    main()
