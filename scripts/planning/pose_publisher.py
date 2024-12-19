#pose_publisher

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import math

def normalize(v):
    return v / np.linalg.norm(v)

def vector_to_quaternion(forward, up=np.array([0,0,1])):
    forward = normalize(forward)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([1,0,0]) 
    z_axis = forward
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    R = np.eye(4)
    R[0:3,0] = x_axis
    R[0:3,1] = y_axis
    R[0:3,2] = z_axis

    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]

    w = math.sqrt(1.0 + r11 + r22 + r33) / 2.0
    x = (r32 - r23) / (4.0 * w)
    y = (r13 - r31) / (4.0 * w)
    z = (r21 - r12) / (4.0 * w)
    return (x, y, z, w)

def rotate_vector(v, axis, angle):
    # Rodrigues' rotation formula: v_rot = v*cos(angle) + (k x v)*sin(angle) + k*(k·v)*(1 - cos(angle))
    k = normalize(axis)
    v = np.array(v)
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    v_rot = v * cos_theta + np.cross(k, v) * sin_theta + k * (np.dot(k, v)) * (1 - cos_theta)
    return v_rot

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        self.publisher = self.create_publisher(PoseStamped, 'target_pose', 10)
        self.timer = self.create_timer(5.0, self.publish_random_pose)
        self.sphere_center = np.array([-0.0304, 0.6437, 0.3910])
        self.sphere_radius = 0.3887

        # 可调偏差角度(度), 若为0则为严格指向球心
        self.orientation_deviation_deg = 5.0  
        
    def generate_random_point_on_upper_hemisphere(self):
        while True:
            direction = np.random.uniform(-1,1,size=3)
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-6:
                direction = direction / norm_dir
                # 上半球, z>0
                if direction[2] > 0:
                    return self.sphere_center + direction * self.sphere_radius

    def publish_random_pose(self):
        random_position = self.generate_random_point_on_upper_hemisphere()
        direction_to_center = self.sphere_center - random_position
        direction_to_center = normalize(direction_to_center)

        # 在 direction_to_center 的方向附近加入一个小偏差
        deviation_rad = math.radians(self.orientation_deviation_deg)
        if deviation_rad > 1e-6:
            # 找一个与direction_to_center垂直的轴
            rot_axis = np.cross(direction_to_center, [0,0,1])
            if np.linalg.norm(rot_axis) < 1e-6:
                rot_axis = np.cross(direction_to_center, [0,1,0])
            rot_axis = normalize(rot_axis)

            # 在[-deviation_rad, deviation_rad]范围内随机一个角度偏差
            angle = (np.random.rand()*2 - 1) * deviation_rad
            # 绕rot_axis旋转direction_to_center
            direction_to_center = rotate_vector(direction_to_center, rot_axis, angle)

        x, y, z, w = vector_to_quaternion(direction_to_center, up=np.array([0,0,1]))

        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_link"
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = float(random_position[0])
        target_pose.pose.position.y = float(random_position[1])
        target_pose.pose.position.z = float(random_position[2])
        target_pose.pose.orientation.x = x
        target_pose.pose.orientation.y = y
        target_pose.pose.orientation.z = z
        target_pose.pose.orientation.w = w

        self.publisher.publish(target_pose)

        # 输出当前位置姿态
        self.get_logger().info("已发布目标位姿：")
        self.get_logger().info(f"位置: x={target_pose.pose.position.x:.4f}, y={target_pose.pose.position.y:.4f}, z={target_pose.pose.position.z:.4f}")
        self.get_logger().info(f"姿态(四元数): x={x:.4f}, y={y:.4f}, z={z:.4f}, w={w:.4f}")

def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
