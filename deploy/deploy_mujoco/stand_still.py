import os
import time
import mujoco
import mujoco.viewer
import numpy as np

def main():
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
    
    # --- Initial joint values for 29-DOF G1 ---
    # These match the values found in mujoco_sim_node.py
    # initial_joint_values = np.array([
    #     -0.413, 0.0, 0.0, 0.807, -0.374, 0.0, -0.413, 0.0, 0.0, 0.807,
    #     -0.374, 0.0, 0.0, 0.0, 0.0, 0.498, 0.3, 0.0, 0.501, 0.0,
    #     0.0, 0.0, 0.498, -0.3, 0.0, 0.501, 0.0, 0.0, 0.0
    # ], dtype=np.float64)
    initial_joint_values = np.array([
        -1.5, 0.0, 0.0, 1.007, 1.52, 0,  # legs
        -0.613, 0.0, 0.0, 1.007, -0.374, 0.0,   
        0.0, 0.0, 0.0,                         # waist 
        0.498,  0.3, 0.0, 0.501, 0.0, 0.0, 0.0,# arms  
        0.498, -0.3, 0.0, 0.501, 0.0, 0.0, 0.0
    ], dtype=np.float64)

    
    num_joints = len(initial_joint_values)
    
    # Set initial positions in qpos
    # Indices 0-6 are the floating base (pos:3, quat:4)
    # Joints start at index 7
    d.qpos[7:7 + num_joints] = initial_joint_values
    
    # PD gains
    kp = 200.0
    kd = 10.0
    
    # Capture initial base pose
    initial_base_pos = d.qpos[0:3].copy()
    initial_base_quat = d.qpos[3:7].copy()
    
    # Launch passive viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("MuJoCo simulation started. Robot is permanently held in its default pose.")
        print("Press Ctrl+C to stop.")
        
        # Set camera to track the robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1  # Pelvis
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- PD Control for Joints ---
            q = d.qpos[7:7 + num_joints]
            v = d.qvel[6:6 + num_joints]
            torques = kp * (initial_joint_values - q) - kd * v
            d.ctrl[:num_joints] = torques
            
            # --- Permanent Stability: Lock base X, Y and Orientation ---
            # We don't lock Z so the robot can touch the ground
            d.qpos[0:2] = initial_base_pos[0:2] # Lock X, Y
            d.qpos[3:7] = initial_base_quat     # Lock Orientation
            
            # Zero out the velocities for the locked dimensions
            d.qvel[0:2] = 0.0 # x, y
            d.qvel[3:6] = 0.0 # orientation velocities
            
            # Step simulation
            mujoco.mj_step(m, d)
            
            # Sync viewer
            viewer.sync()
            
            # Real-time synchronization
            elapsed = time.time() - step_start
            if elapsed < simulation_dt:
                time.sleep(simulation_dt - elapsed)

if __name__ == "__main__":
    main()
