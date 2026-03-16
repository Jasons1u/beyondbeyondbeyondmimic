import os
import time
import argparse
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np

def parse_joint_names_from_xml(xml_path):
    joint_names = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for joint in root.iter("joint"):
        name = joint.attrib.get("name", None)
        if name is not None:
            joint_names.append(name)
    if joint_names and joint_names[0] == "floating_base_joint":
        joint_names.pop(0)
    return joint_names

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--motion", type=str, default="artifacts/traj_jump4:v0/motion.npz", help="Path to motion NPZ file")
    parser.add_argument("--motion", type=str, default="artifacts/srb_jump_1:v0/motion_edited.npz", help="Path to motion NPZ file")
    args = parser.parse_args()

    # Set environment variable if not already set
    if "G1_RL_ROOT_DIR" not in os.environ:
        os.environ["G1_RL_ROOT_DIR"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
    xml_path = os.path.join(G1_RL_ROOT_DIR, "g1", "scene_29dof.xml")
    
    if not os.path.exists(xml_path):
        print(f"Error: XML path {xml_path} does not exist.")
        return

    # Load model and data
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # Simulation settings
    simulation_dt = 0.002
    m.opt.timestep = simulation_dt

    # Load motion
    motion_path = os.path.join(G1_RL_ROOT_DIR, args.motion)
    if not os.path.exists(motion_path):
        print(f"Error: Motion file {motion_path} does not exist.")
        return
    
    motion_data = np.load(motion_path)
    fps = motion_data["fps"]
    if isinstance(fps, np.ndarray):
        fps = fps.item()
    frame_dt = 1.0 / fps
    
    joint_pos_traj = motion_data["joint_pos"]
    body_pos_w = motion_data["body_pos_w"]
    body_quat_w = motion_data["body_quat_w"]
    num_frames = joint_pos_traj.shape[0]

    print(f"Loaded motion: {num_frames} frames at {fps} FPS")

    # Joint mapping (IsaacLab/ONNX -> MuJoCo)
    # Read joints directly from the MuJoCo model
    xml_joint_names = []
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "floating_base_joint":
            xml_joint_names.append(name)
    onnx_joint_names = [
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 
        'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 
        'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 
        'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 
        'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
        'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 
        'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
    ]
    
    xml_to_onnx = []
    for name in xml_joint_names:
        idx = onnx_joint_names.index(name) if name in onnx_joint_names else -1
        xml_to_onnx.append(idx)

    num_joints = len(onnx_joint_names)
    
    # Launch passive viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("MuJoCo simulation started. Playing motion kinematically.")
        print("Press Ctrl+C to stop.")
        
        # Set camera to track the robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1  # Pelvis
        
        frame_idx = 0
        last_frame_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # Kinematically set the pose for the current frame
            # 1. Floating base
            root_pos = body_pos_w[frame_idx, 0] # Body 0 is pelvis
            root_quat = body_quat_w[frame_idx, 0]
            
            d.qpos[0:3] = root_pos
            d.qpos[3:7] = root_quat
            
            # 2. Joints
            isaac_joints = joint_pos_traj[frame_idx]
            mujoco_joints = np.zeros(num_joints)
            for xml_idx, onnx_idx in enumerate(xml_to_onnx):
                if onnx_idx >= 0 and onnx_idx < len(isaac_joints):
                    mujoco_joints[xml_idx] = isaac_joints[onnx_idx]
            
            d.qpos[7:7+num_joints] = mujoco_joints
            
            # Zero out velocities since we are doing kinematic replay
            d.qvel[:] = 0.0
            
            # We don't step physics, we just update kinematics to visualize
            mujoco.mj_kinematics(m, d)
            
            # Sync viewer
            viewer.sync()
            
            # Advance frame based on time
            elapsed = time.time() - last_frame_time
            if elapsed >= frame_dt:
                frame_idx = (frame_idx + 1) % num_frames
                last_frame_time = time.time()
                
            # Keep loop running at a reasonable rate
            time.sleep(0.002)

if __name__ == "__main__":
    main()
