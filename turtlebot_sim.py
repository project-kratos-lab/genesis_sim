import genesis as gs
from gs_ros.gs_ros_bridge import GsRosBridge
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from rclpy.executors import SingleThreadedExecutor
import re
import os
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

        # Resolve namespace from config so joint names match the URDF/bridge
        cfg_path = os.path.join(os.path.dirname(__file__), "turtlebot_config.yaml")
        self.namespace = self._read_namespace_from_config(cfg_path)

        # Continuously publish joint commands at 50 Hz so the bridge always
        # receives fresh commands and keeps sensors (LiDAR, camera) active.
        self.timer = self.create_timer(1.0 / 50.0, self._publish_joint_commands)

    def _read_namespace_from_config(self, path):
        try:
            import yaml
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            ns = cfg.get("robots", {}).get("turtlebot", {}).get("namespace", "")
            return ns or ""
        except Exception:
            # Fallback: simple regex search for 'namespace: <value>'
            try:
                with open(path, "r") as f:
                    txt = f.read()
                m = re.search(r"namespace:\s*[\"']?([A-Za-z0-9_\-/]+)[\"']?", txt)
                return m.group(1) if m else ""
            except Exception:
                return ""

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
        left_name = f"{self.namespace}wheel_left_joint" if self.namespace else "wheel_left_joint"
        right_name = f"{self.namespace}wheel_right_joint" if self.namespace else "wheel_right_joint"
        js.name = [left_name, right_name]
        
        # JointState field is 'velocity' (singular) and expects a list
        js.velocity = [self.w_l, self.w_r]
        
        # Publish to the corrected command topics
        self.jstate_pub.publish(js)
        self.jstate_pub_alt.publish(js)


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
    
    # Configure executor to spin all nodes (including sensor nodes)
    print("ROS Bridge spin nodes:", [n.get_name() for n in ros_bridge.all_nodes_to_spin])
    
    executor = SingleThreadedExecutor()
    executor.add_node(teleop_node)
    
    # Add all bridge-managed nodes to executor
    for node in ros_bridge.all_nodes_to_spin:
        executor.add_node(node)
    ros_bridge.all_nodes_to_spin = []
    
    # Add simulation interface to executor if present
    if hasattr(ros_bridge, "simulation_interface"):
        executor.add_node(ros_bridge.simulation_interface)
        delattr(ros_bridge, "simulation_interface")

    # 5. Simulation loop - bridge.step() handles physics
    try:
        while rclpy.ok():
            ros_bridge.step()
            executor.spin_once(timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy all nodes in executor
        for node in executor.get_nodes():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()