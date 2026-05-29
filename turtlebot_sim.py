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

class CmdVelToJoints(Node):
    def __init__(self):
        super().__init__("cmd_vel_to_joints")
        self.sub = self.create_subscription(Twist, "cmd_vel", self.cmd_vel_callback, 10)
        # Publish wheel velocities as JointState (has 'velocity' field) which
        # the bridge expects and won't crash on.
        self.jstate_pub = self.create_publisher(JointState, "/turtlebot/joint_states", 10)
        self.jstate_pub_alt = self.create_publisher(JointState, "/robot/joint_states", 10)
        
        # Turtlebot3 Burger specs
        self.wheel_separation = 0.160
        self.wheel_radius = 0.033

        # Resolve namespace from config so joint names match the URDF/bridge
        cfg_path = os.path.join(os.path.dirname(__file__), "turtlebot_config.yaml")
        self.namespace = self._read_namespace_from_config(cfg_path)

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
        # v = (v_r + v_l) / 2
        # w = (v_r - v_l) / w_sep
        # v_r = w_wheel_r * w_rad
        
        v_r = v + (w * self.wheel_separation / 2.0)
        v_l = v - (w * self.wheel_separation / 2.0)
        
        # Convert to joint angular velocity (rad/s)
        w_r = v_r / self.wheel_radius
        w_l = v_l / self.wheel_radius
        
        # Publish JointState with velocities (bridge uses 'velocity' attribute)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        left_name = f"{self.namespace}wheel_left_joint" if self.namespace else "wheel_left_joint"
        right_name = f"{self.namespace}wheel_right_joint" if self.namespace else "wheel_right_joint"
        js.name = [left_name, right_name]
        # JointState field is 'velocity' (singular) and expects a list
        js.velocity = [w_l, w_r]
        try:
            self.get_logger().debug(f"Publishing joint_state: {js.name} vel={js.velocity}")
        except Exception:
            pass
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
    
    executor = SingleThreadedExecutor()
    executor.add_node(teleop_node)
    executor.add_node(ros_node)

    # 5. Simulation loop - bridge.step() handles physics + ROS spinning
    try:
        while rclpy.ok():
            ros_bridge.step()
            executor.spin_once(timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        teleop_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
