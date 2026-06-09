import genesis as gs
from gs_ros.gs_ros_bridge import GsRosBridge
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from rclpy.executors import SingleThreadedExecutor
import tf2_ros
import re
import os
import subprocess
import math
import numpy as np
# 1. Strip 'enable_interaction' from ViewerOptions
original_viewer_init = gs.options.ViewerOptions.__init__
def patched_viewer_init(self, *args, **kwargs):
    kwargs.pop("enable_interaction", None)
    original_viewer_init(self, *args, **kwargs)
gs.options.ViewerOptions.__init__ = patched_viewer_init

# 2. Strip 'avatar_options' from Scene
original_scene_init = gs.Scene.__init__
def patched_scene_init(self, *args, **kwargs):
    kwargs.pop("avatar_options", None)
    original_scene_init(self, *args, **kwargs)
gs.Scene.__init__ = patched_scene_init
# 3. Strip explicit 'dt=None' from RigidOptions
original_rigid_init = gs.options.RigidOptions.__init__
def patched_rigid_init(self, *args, **kwargs):
    if "dt" in kwargs and kwargs["dt"] is None:
        kwargs.pop("dt")  # Removes None so Genesis uses its safe default
    original_rigid_init(self, *args, **kwargs)
gs.options.RigidOptions.__init__ = patched_rigid_init
original_lidar_init = gs.sensors.Lidar.__init__
def patched_lidar_init(self, *args, **kwargs):
    # Genesis 1.1.0 strictly requires this to be > 0
    if kwargs.get("debug_sphere_radius", 0.0) <= 0.0:
        kwargs["debug_sphere_radius"] = 0.01
    original_lidar_init(self, *args, **kwargs)
gs.sensors.Lidar.__init__ = patched_lidar_init

class CmdVelToJoints(Node):
    def __init__(self):
        super().__init__("cmd_vel_to_joints")
        self.sub = self.create_subscription(Twist, "cmd_vel", self.cmd_vel_callback, 10)
        
        # FIXED: Publish wheel velocities to 'joint_commands' instead of 'joint_states'
        self.jstate_pub = self.create_publisher(JointState, "/turtlebot/joint_commands", 10)
        self.jstate_pub_alt = self.create_publisher(JointState, "/robot/joint_commands", 10)
        
        # Turtlebot3 Waffle specs
        self.wheel_separation = 0.287
        self.wheel_radius = 0.033

        # Track last commanded wheel velocities (default to zero / stopped)
        self.w_l = 0.0
        self.w_r = 0.0

        # Continuously publish joint commands at 50 Hz so the bridge always
        # receives fresh commands and keeps sensors (LiDAR, camera) active.
        self.timer = self.create_timer(1.0 / 50.0, self._publish_joint_commands)

    def cmd_vel_callback(self, msg):
        v = msg.linear.x
        w = msg.angular.z
        
        # Differential drive kinematics
        v_r = v + (w * self.wheel_separation / 2.0)
        v_l = v - (w * self.wheel_separation / 2.0)
        
        # Convert to joint angular velocity (rad/s)
        self.w_r = v_r / self.wheel_radius
        self.w_l = v_l / self.wheel_radius

    def _publish_joint_commands(self):
        """Publish current wheel velocities at a fixed rate, even when idle."""
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        # Use standard (un-prefixed) joint names matching the updated URDF
        js.name = ["wheel_left_joint", "wheel_right_joint"]
        
        # JointState field is 'velocity' (singular) and expects a list
        js.velocity = [self.w_l, self.w_r]
        
        # Publish to the corrected command topics
        self.jstate_pub.publish(js)
        self.jstate_pub_alt.publish(js)


class OdometryPublisher(Node):
    """Publishes odom->base_footprint TF and /odom topic from Genesis ground-truth pose."""

    def __init__(self, ros_bridge, robot_name="turtlebot"):
        super().__init__("odometry_publisher")
        self.ros_bridge = ros_bridge
        self.robot_name = robot_name
        self.robot = None  # Will be set after scene is built

        # TF broadcaster for odom -> base_footprint
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Odometry publisher
        self.odom_pub = self.create_publisher(Odometry, "/odom", 50)

        # Track previous pose for velocity estimation
        self.prev_pos = None
        self.prev_yaw = None
        self.prev_time = None

        # Publish at 50 Hz
        self.timer = self.create_timer(1.0 / 50.0, self._publish_odom)

    def _get_robot(self):
        """Lazily get the robot entity from the bridge after scene is built."""
        if self.robot is None:
            try:
                entities = self.ros_bridge.entities_info
                if self.robot_name in entities:
                    self.robot = entities[self.robot_name]["entity_attr"]
            except Exception:
                pass
        return self.robot

    def _publish_odom(self):
        """Read ground-truth pose from Genesis and publish TF + Odometry."""
        robot = self._get_robot()
        if robot is None:
            return

        try:
            # Get position [x, y, z] from Genesis (world frame)
            pos = robot.get_pos().detach().cpu().numpy().flatten()
            # Get quaternion [w, x, y, z] from Genesis
            quat_gs = robot.get_quat().detach().cpu().numpy().flatten()
        except Exception:
            return

        # Convert Genesis quat (w, x, y, z) -> ROS quat (x, y, z, w)
        qx, qy, qz, qw = quat_gs[1], quat_gs[2], quat_gs[3], quat_gs[0]

        now = self.get_clock().now().to_msg()

        # --- Broadcast TF: odom -> base_footprint ---
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(t)

        # --- Compute velocity estimate ---
        cur_time = now.sec + now.nanosec * 1e-9
        # Extract yaw from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        vx, vy, vz = 0.0, 0.0, 0.0
        angular_z = 0.0
        if self.prev_pos is not None and self.prev_time is not None:
            dt = cur_time - self.prev_time
            if dt > 1e-6:
                # World-frame velocity
                dx = pos[0] - self.prev_pos[0]
                dy = pos[1] - self.prev_pos[1]
                # Transform to body frame
                vx = (dx * math.cos(-yaw) - dy * math.sin(-yaw)) / dt
                vy = (dx * math.sin(-yaw) + dy * math.cos(-yaw)) / dt
                # Angular velocity
                dyaw = yaw - self.prev_yaw
                # Normalize to [-pi, pi]
                dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                angular_z = dyaw / dt

        self.prev_pos = pos.copy()
        self.prev_yaw = yaw
        self.prev_time = cur_time

        # --- Publish Odometry message ---
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_footprint"

        odom_msg.pose.pose.position.x = float(pos[0])
        odom_msg.pose.pose.position.y = float(pos[1])
        odom_msg.pose.pose.position.z = float(pos[2])
        odom_msg.pose.pose.orientation.x = float(qx)
        odom_msg.pose.pose.orientation.y = float(qy)
        odom_msg.pose.pose.orientation.z = float(qz)
        odom_msg.pose.pose.orientation.w = float(qw)

        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.angular.z = angular_z

        self.odom_pub.publish(odom_msg)


def launch_robot_state_publisher():
    """Launch robot_state_publisher as a subprocess to publish URDF TF frames."""
    urdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "turtlebot3_description", "urdf", "turtlebot3_waffle.urdf"
    )

    # Read the URDF file content
    with open(urdf_path, "r") as f:
        urdf_content = f.read()

    # Launch robot_state_publisher with the URDF as a parameter
    # It will subscribe to /turtlebot/joint_states for wheel transforms
    # and publish static TF for all fixed joints
    cmd = [
        "ros2", "run", "robot_state_publisher", "robot_state_publisher",
        "--ros-args",
        "-p", f"robot_description:={urdf_content}",
        "-r", "joint_states:=/turtlebot/joint_states",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[turtlebot_sim] robot_state_publisher started (PID {proc.pid})")
    return proc


def main():
    # 1. Initialize Genesis with performance mode enabled
    gs.init(backend=gs.gpu, performance_mode=True)

    # 2. Initialize ROS 2
    rclpy.init()
    ros_node = Node("turtlebot_sim")
    teleop_node = CmdVelToJoints()

    # 3. Create the GsRosBridge with config file
    ros_bridge = GsRosBridge(
        ros_node=ros_node,
        file_path="turtlebot_config.yaml",
    )

    # 4. Build the scene
    ros_bridge.build()

    # 5. Launch robot_state_publisher for URDF TF frames
    rsp_proc = launch_robot_state_publisher()

    # 6. Create odometry publisher (odom -> base_footprint TF + /odom topic)
    odom_node = OdometryPublisher(ros_bridge, robot_name="turtlebot")

    # Configure executor to spin all nodes (including sensor nodes)
    print("ROS Bridge spin nodes:", [n.get_name() for n in ros_bridge.all_nodes_to_spin])
    
    executor = SingleThreadedExecutor()
    executor.add_node(teleop_node)
    executor.add_node(odom_node)
    
    # Add all bridge-managed nodes to executor
    for node in ros_bridge.all_nodes_to_spin:
        executor.add_node(node)
    ros_bridge.all_nodes_to_spin = []
    
    # Add simulation interface to executor if present
    if hasattr(ros_bridge, "simulation_interface"):
        executor.add_node(ros_bridge.simulation_interface)
        delattr(ros_bridge, "simulation_interface")

    # 7. Simulation loop - bridge.step() handles physics
    try:
        while rclpy.ok():
            ros_bridge.step()
            executor.spin_once(timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        # Terminate robot_state_publisher
        rsp_proc.terminate()
        rsp_proc.wait()
        # Destroy all nodes in executor
        for node in executor.get_nodes():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()