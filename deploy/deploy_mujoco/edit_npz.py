import os
import argparse
import numpy as np
import mujoco


# Isaac/ONNX joint order (same as joint_pos in the NPZ)
ONNX_JOINT_NAMES = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint',
    'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
    'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
    'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
]

# Mapping from NPZ body index (Isaac, 37 bodies) -> MuJoCo body index (30 bodies, 0-indexed after skipping world)
# Built by matching positions between Isaac NPZ and MuJoCo FK output
NPZ_TO_MJ_BODY = {
    0: 0,    # pelvis
    1: 1,    # left_hip_pitch_link
    2: 0,    # pelvis (Isaac duplicate)
    3: 7,    # right_hip_pitch_link
    4: 0,    # pelvis (Isaac duplicate)
    5: 2,    # left_hip_roll_link
    6: 8,    # right_hip_roll_link
    7: 14,   # waist_roll_link
    8: 3,    # left_hip_yaw_link
    9: 9,    # right_hip_yaw_link
    10: 15,  # torso_link
    11: 4,   # left_knee_link
    12: 10,  # right_knee_link
    13: 0,   # pelvis (Isaac duplicate)
    14: 16,  # left_shoulder_pitch_link
    15: 0,   # pelvis (Isaac duplicate)
    16: 23,  # right_shoulder_pitch_link
    17: 5,   # left_ankle_pitch_link
    18: 11,  # right_ankle_pitch_link
    19: 17,  # left_shoulder_roll_link
    20: 24,  # right_shoulder_roll_link
    21: 6,   # left_ankle_roll_link
    22: 12,  # right_ankle_roll_link
    23: 18,  # left_shoulder_yaw_link
    24: 25,  # right_shoulder_yaw_link
    25: 6,   # left foot site (closest: left_ankle_roll_link)
    26: 12,  # right foot site (closest: right_ankle_roll_link)
    27: 19,  # left_elbow_link
    28: 26,  # right_elbow_link
    29: 20,  # left_wrist_roll_link
    30: 27,  # right_wrist_roll_link
    31: 21,  # left_wrist_pitch_link
    32: 28,  # right_wrist_pitch_link
    33: 22,  # left_wrist_yaw_link
    34: 29,  # right_wrist_yaw_link
    35: 22,  # left hand site (closest: left_wrist_yaw_link)
    36: 29,  # right hand site (closest: right_wrist_yaw_link)
}


def get_xml_joint_names(xml_path):
    """Get joint names from MuJoCo model in order, excluding floating base."""
    m = mujoco.MjModel.from_xml_path(xml_path)
    joint_names = []
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "floating_base_joint":
            joint_names.append(name)
    return joint_names


def mujoco_to_isaac_joints(arr, xml_joint_names, onnx_joint_names):
    """Convert joint array from MuJoCo/XML order to Isaac/ONNX order."""
    out = np.zeros_like(arr)
    for onnx_idx, onnx_name in enumerate(onnx_joint_names):
        try:
            xml_idx = xml_joint_names.index(onnx_name)
            out[onnx_idx] = arr[xml_idx]
        except ValueError:
            pass
    return out


def compute_fk_and_convert_to_isaac(xml_path, root_pos, root_quat, joint_values_isaac, num_npz_bodies=37):
    """
    Run MuJoCo FK with the given joint values (in Isaac order), then reorder
    the resulting body poses from MuJoCo body order to Isaac/NPZ body order.

    Returns body_pos (num_npz_bodies, 3), body_quat (num_npz_bodies, 4) in Isaac order.
    """
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Get XML joint names
    xml_joint_names = []
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "floating_base_joint":
            xml_joint_names.append(name)

    # Set floating base
    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat

    # Set joints: convert from Isaac/ONNX order to MuJoCo/XML order
    for xml_idx, xml_name in enumerate(xml_joint_names):
        onnx_idx = ONNX_JOINT_NAMES.index(xml_name)
        d.qpos[7 + xml_idx] = joint_values_isaac[onnx_idx]

    d.qvel[:] = 0.0
    mujoco.mj_kinematics(m, d)

    # MuJoCo body poses (skip world body 0)
    mj_body_pos = d.xpos[1:].copy().astype(np.float32)    # (30, 3)
    mj_body_quat = d.xquat[1:].copy().astype(np.float32)  # (30, 4)

    # Reorder from MuJoCo to Isaac/NPZ body order
    isaac_body_pos = np.zeros((num_npz_bodies, 3), dtype=np.float32)
    isaac_body_quat = np.zeros((num_npz_bodies, 4), dtype=np.float32)
    isaac_body_quat[:, 0] = 1.0  # default identity quaternion

    for npz_idx, mj_idx in NPZ_TO_MJ_BODY.items():
        if npz_idx < num_npz_bodies and mj_idx < len(mj_body_pos):
            isaac_body_pos[npz_idx] = mj_body_pos[mj_idx]
            isaac_body_quat[npz_idx] = mj_body_quat[mj_idx]

    return isaac_body_pos, isaac_body_quat


def main():
    parser = argparse.ArgumentParser(description="Edit NPZ motion files")
    parser.add_argument("--motion", type=str, default="artifacts/srb_jump_1:v0/motion.npz", help="Path to motion NPZ file")
    args = parser.parse_args()

    # Resolve path
    if "G1_RL_ROOT_DIR" not in os.environ:
        os.environ["G1_RL_ROOT_DIR"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
    motion_path = os.path.join(G1_RL_ROOT_DIR, args.motion)
    xml_path = os.path.join(G1_RL_ROOT_DIR, "g1", "scene_29dof.xml")

    if not os.path.exists(motion_path):
        print(f"Error: Motion file {motion_path} does not exist.")
        return

    # Load motion
    motion_data = np.load(motion_path)

    # Print all topics (keys) with their shape and dtype
    print(f"Motion file: {motion_path}")
    print(f"{'Key':<20} {'Shape':<25} {'Dtype':<10}")
    print("-" * 55)
    for key in motion_data.keys():
        arr = motion_data[key]
        print(f"{key:<20} {str(arr.shape):<25} {str(arr.dtype):<10}")

    # Joint orderings
    xml_joint_names = get_xml_joint_names(xml_path)

    # Initial joint values in MuJoCo/XML order (left leg, right leg, waist, left arm, right arm)
    initial_joint_values_mujoco = np.array([
        -0.372, 0.00753, -0.0163, 0.712, -0.39, -0.0185,  # left leg
        -0.372, 0.00753, -0.0163, 0.712, -0.39, -0.0185,  # right leg
        0.0, 0.0, 0.0,                                      # waist
        0.471,  0.3, 0.0, 0.51, 0.0, 0.0, 0.0,             # left arm
        0.471, -0.3, 0.0, 0.51, 0.0, 0.0, 0.0              # right arm
    ], dtype=np.float32)

    # Convert to Isaac/ONNX order for the NPZ
    initial_joint_values_isaac = mujoco_to_isaac_joints(
        initial_joint_values_mujoco, xml_joint_names, ONNX_JOINT_NAMES
    )

    print(f"\nInitial joints (MuJoCo order): {initial_joint_values_mujoco}")
    print(f"Initial joints (Isaac order):  {initial_joint_values_isaac}")

    num_new_frames = 20
    num_frames = motion_data["joint_pos"].shape[0]
    num_bodies_npz = motion_data["body_pos_w"].shape[1]

    # Use the original frame 0 root pose as the starting root pose
    root_pos = motion_data["body_pos_w"][0, 0]   # pelvis XYZ
    root_quat = motion_data["body_quat_w"][0, 0]  # pelvis quaternion

    # Run FK with Isaac-order joints, get body poses in Isaac/NPZ body order
    fk_body_pos, fk_body_quat = compute_fk_and_convert_to_isaac(
        xml_path, root_pos, root_quat, initial_joint_values_isaac, num_bodies_npz
    )

    # Build new arrays by prepending frames
    # joint_pos: initial values in Isaac order
    new_joint_pos = np.vstack([
        np.tile(initial_joint_values_isaac, (num_new_frames, 1)),
        motion_data["joint_pos"]
    ])

    # joint_vel: zeros
    new_joint_vel = np.vstack([
        np.zeros((num_new_frames, motion_data["joint_vel"].shape[1]), dtype=np.float32),
        motion_data["joint_vel"]
    ])

    # body_pos_w: FK-computed, in Isaac body order
    new_body_pos_w = np.concatenate([
        np.tile(fk_body_pos[np.newaxis, :, :], (num_new_frames, 1, 1)),
        motion_data["body_pos_w"]
    ], axis=0)

    # body_quat_w: FK-computed, in Isaac body order
    new_body_quat_w = np.concatenate([
        np.tile(fk_body_quat[np.newaxis, :, :], (num_new_frames, 1, 1)),
        motion_data["body_quat_w"]
    ], axis=0)

    # body_lin_vel_w: zeros for new frames
    new_body_lin_vel_w = np.concatenate([
        np.zeros((num_new_frames, num_bodies_npz, 3), dtype=np.float32),
        motion_data["body_lin_vel_w"]
    ], axis=0)

    # body_ang_vel_w: zeros for new frames
    new_body_ang_vel_w = np.concatenate([
        np.zeros((num_new_frames, num_bodies_npz, 3), dtype=np.float32),
        motion_data["body_ang_vel_w"]
    ], axis=0)

    # Save edited file
    base, ext = os.path.splitext(motion_path)
    output_path = base + "_edited" + ext

    np.savez(
        output_path,
        fps=motion_data["fps"],
        joint_pos=new_joint_pos,
        joint_vel=new_joint_vel,
        body_pos_w=new_body_pos_w,
        body_quat_w=new_body_quat_w,
        body_lin_vel_w=new_body_lin_vel_w,
        body_ang_vel_w=new_body_ang_vel_w,
    )

    print(f"\nSaved edited motion to: {output_path}")
    print(f"Original frames: {num_frames}, New frames: {num_frames + num_new_frames}")

    # Verify
    edited = np.load(output_path)
    print(f"\nVerification:")
    print(f"{'Key':<20} {'Shape':<25} {'Dtype':<10}")
    print("-" * 55)
    for key in edited.keys():
        arr = edited[key]
        print(f"{key:<20} {str(arr.shape):<25} {str(arr.dtype):<10}")


if __name__ == "__main__":
    main()
