import argparse
import os

# Update the viewer at 30Hz
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
# from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray, Float64



class MujocoSimNode(Node):
    def __init__(self, config_file):

        super().__init__("mujoco_sim_node")
        self.config_file = config_file
        self.publisher = self.create_publisher(Float32MultiArray, "sensor_data", 10)
        self.robot_position_pub = self.create_publisher(
            Float32MultiArray, "robot_position", 10
        )
        self.time_pub = self.create_publisher(Float64, "sim_time", 10)
        
        self.height_map_pub = self.create_publisher(Float32MultiArray, "height_map", 10)
        self.base_lin_vel_pub = self.create_publisher(
            Float32MultiArray, "base_lin_vel", 10
        )
        # Publisher to signal RL policy to enable control (Float64: 0.0/1.0)
        self.control_enable_pub = self.create_publisher(Float64, "control_enable", 10)

        self.ball_position_pub = self.create_publisher(
            Float32MultiArray, "ball_position", 10
        )

        self.action_sub = self.create_subscription(
            Float32MultiArray, "action", self.action_callback, 10
        )
        self.initial_pose_sub = self.create_subscription(
            Float32MultiArray, "initial_pose", self.initial_pose_callback, 10
        )
        self.action = None
        self.timer = self.create_timer(0.0, self.sim_step)  # As fast as possible
        self.num_joints = 29
        self.load_config()
        self.init_sim()
        # control enable published once after hold period
        self._control_enable_sent = False
        # duration to hold robot upright before enabling policy (seconds)
        self._hold_upright_duration = 4.0
        # whether we received the initial pose from policy to start the hold timer
        self._initial_pose_received = False
        self._hold_start_time = 0.0
        # one-time prints
        self._hold_printed = False
        self._released_printed = False
        
        # Ball spawn configuration
        self.ball_spawn_time = 0.0  # Spawn ball at t=0.0s
        self.ball_position = np.array([0.4, 0.2, 1.0])  # x, y, z position
        self.ball_spawned = False

    # self.sim_time = 0.0

    def load_config(self):
        G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
        with open(f"{G1_RL_ROOT_DIR}/deploy/configs/{self.config_file}") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.simulation_dt = config["simulation_dt"]

        self.xml_path = config["xml_path"].replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)


    def init_sim(self):
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        self.target_dof_pos = np.zeros(self.num_joints)
        self.target_dof_p = np.zeros(self.num_joints)
        self.target_dof_d = np.zeros(self.num_joints)

        # --- Initialize robot joints to provided values ---
        self.initial_joint_values = np.array([
            -0.413, 0.0, 0.0, 0.807, -0.374, 0.0, -0.413, 0.0, 0.0, 0.807,
            -0.374, 0.0, 0.0, 0.0, 0.0, 0.498, 0.3, 0.0, 0.501, 0.0,
            0.0, 0.0, 0.498, -0.3, 0.0, 0.501, 0.0, 0.0, 0.0
        ], dtype=np.float64)
        self.d.qpos[7:7 + self.num_joints] = self.initial_joint_values
        # Optionally, also set target positions to match initial
        self.target_dof_pos = self.initial_joint_values.copy()
        
        # Find ball body and joint indices
        try:
            self.ball_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "ball")
            ball_joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
            self.ball_joint_qpos_adr = self.m.jnt_qposadr[ball_joint_id]
            self.ball_joint_qvel_adr = self.m.jnt_dofadr[ball_joint_id]
            self.get_logger().info(f"Ball found: body_id={self.ball_body_id}, qpos_adr={self.ball_joint_qpos_adr}")
        except Exception as e:
            self.get_logger().warning(f"Ball body/joint not found in model: {e}")
            self.ball_body_id = None

        # Launch the viewer in passive mode
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        self.viewer_sync_interval = 1.0 / 30.0  # 30Hz
        self._last_viewer_sync_time = 0.0


    def action_callback(self, msg):
        # Expect msg.data to be [position, p, d] 
        arr = np.array(msg.data, dtype=np.float32)
        n = arr.shape[0] // 3
        self.target_dof_pos = arr[:n]
        self.target_dof_p = arr[n : 2 * n]
        self.target_dof_d = arr[2 * n : 3 * n]

    def initial_pose_callback(self, msg):
        """
        Callback to set the initial pose of the robot based on policy neutral state.
        Starts the hold timer when first received.
        """
        new_initial_values = np.array(msg.data, dtype=np.float64)
        if len(new_initial_values) == self.num_joints:
            if not self._initial_pose_received:
                self.get_logger().info(f"[MujocoSimNode] Received first initial pose. Starting {self._hold_upright_duration}s hold timer.")
                self._initial_pose_received = True
                self._hold_start_time = self.d.time
            
            # Update the pose if we haven't enabled control yet
            if not self._control_enable_sent:
                self.initial_joint_values = new_initial_values
                self.d.qpos[7:7 + self.num_joints] = self.initial_joint_values
                self.target_dof_pos = self.initial_joint_values.copy()
                # Also reset velocities to zero for stability
                self.d.qvel[6:6 + self.num_joints] = 0.0
        else:
            self.get_logger().warn(f"[MujocoSimNode] Initial pose size mismatch: expected {self.num_joints}, got {len(new_initial_values)}")

    def sim_step(self):
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1  # ID of the robot's main body
        start_time = time.time()  # Start timing at the very beginning of the step
        
        # Spawn ball at prescribed time and position
        if not self.ball_spawned and self.d.time >= self.ball_spawn_time and self.ball_body_id is not None:
            # Set ball position (qpos for free joint: x, y, z, qw, qx, qy, qz)
            self.d.qpos[self.ball_joint_qpos_adr:self.ball_joint_qpos_adr+3] = self.ball_position
            self.d.qpos[self.ball_joint_qpos_adr+3:self.ball_joint_qpos_adr+7] = [1, 0, 0, 0]  # identity quaternion
            # Zero initial velocity
            self.d.qvel[self.ball_joint_qvel_adr:self.ball_joint_qvel_adr+6] = 0.0
            self.ball_spawned = True
            self.get_logger().info(f"Ball spawned at position {self.ball_position} at t={self.d.time:.3f}s")
        
        # Determine if we are in the "hold" phase (either waiting for pose or within hold duration)
        is_holding = True
        if self._initial_pose_received:
            elapsed_hold = self.d.time - self._hold_start_time
            if elapsed_hold >= self._hold_upright_duration:
                is_holding = False

        # Force robot upright during hold
        if is_holding:
            if not self._hold_printed:
                print(f"[MujocoSimNode] Holding robot upright for {self._hold_upright_duration}s", flush=True)
                self._hold_printed = True
            # Force pelvis/base position and orientation
            self.d.qpos[0] = 0.0
            self.d.qpos[1] = 0.0
            try:
                self.d.qpos[3] = 1.0
                self.d.qpos[4] = 0.0
                self.d.qpos[5] = 0.0
                self.d.qpos[6] = 0.0
            except Exception:
                pass
            # zero base linear (x, y) and angular velocities, but let z velocity drop naturally
            try:
                self.d.qvel[0] = 0.0
                self.d.qvel[1] = 0.0
                # don't zero Z velocity (index 2)
                self.d.qvel[3] = 0.0
                self.d.qvel[4] = 0.0
                self.d.qvel[5] = 0.0
            except Exception:
                pass

            # If target_dof_pos is not yet set by an action, use the initial pose from policy
            if np.all(self.target_dof_pos == 0):
                self.target_dof_pos = self.initial_joint_values

        self.d.ctrl[: self.num_joints] = self.target_dof_pos

        mujoco.mj_step(self.m, self.d)

        # Prepare sensor data
        qj = self.d.sensordata[: self.num_joints]  # [self.joint_ids]
        dqj = self.d.sensordata[
            self.num_joints : 2 * self.num_joints
        ]  # [self.joint_ids]
        tqj = self.d.sensordata[2 * self.num_joints : 3 * self.num_joints]
        quat = self.d.sensordata[3 * self.num_joints : 3 * self.num_joints + 4]
        omega = self.d.sensordata[3 * self.num_joints + 4 : 3 * self.num_joints + 7]
        # Compose sensor data
        sensor_data = np.concatenate([qj, dqj, quat, omega])
        msg = Float32MultiArray()
        msg.data = sensor_data.tolist()
        self.publisher.publish(msg)
        robot_position = np.array(
            self.d.xpos[1]
        )  # Get the robot's base position in world coordinates
        robot_position_msg = Float32MultiArray()
        robot_position_msg.data = robot_position.tolist()
        self.robot_position_pub.publish(robot_position_msg)
        tmsg = Float64()
        tmsg.data = self.d.time  # Use actual mujoco sim time
        self.time_pub.publish(tmsg)

        # Publish a single control_enable flag when hold period ends
        if (self._initial_pose_received and not self._control_enable_sent) and (self.d.time - self._hold_start_time >= self._hold_upright_duration):
            flag = Float64()
            flag.data = 1.0
            self.control_enable_pub.publish(flag)
            self._control_enable_sent = True
            print(f"[MujocoSimNode] Published control_enable=1.0 at sim_time={self.d.time:.3f}", flush=True)

        # Publish base linear velocity (from first 3 elements of qvel)
        base_lin_vel = self.d.qvel[0:3]
        base_lin_vel_msg = Float32MultiArray()
        base_lin_vel_msg.data = base_lin_vel.tolist()
        self.base_lin_vel_pub.publish(base_lin_vel_msg)

        # Publish ball position to RL policy
        if self.ball_body_id is not None:
            ball_pos = self.d.xpos[self.ball_body_id]
            ball_pos_msg = Float32MultiArray()
            ball_pos_msg.data = ball_pos.tolist()
            self.ball_position_pub.publish(ball_pos_msg)

        # --- Synchronize simulation time with real time ---
        sim_dt = self.simulation_dt
        elapsed = time.time() - start_time
        if elapsed < sim_dt:
            time.sleep(sim_dt - elapsed)
        if (
            self.viewer.is_running()
            and (time.time() - self._last_viewer_sync_time) >= self.viewer_sync_interval
        ):
            self.viewer.sync()
            self._last_viewer_sync_time = time.time()
        # --- end synchronization ---


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(
        description="Launch MujocoSimNode and RLPolicyNode with a given config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file name (e.g., g1_29dof.yaml)",
    )
    args = parser.parse_args()
    node = MujocoSimNode(config_file=args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Attempt to close the mujoco viewer gracefully
        if hasattr(node, "viewer") and node.viewer is not None:
            try:
                node.viewer.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
