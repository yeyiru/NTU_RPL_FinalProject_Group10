#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R

class HemisphereTargetPublisher(Node):
    def __init__(self):
        super().__init__('hemisphere_target_publisher')
        
        self.publisher_ = self.create_publisher(PoseStamped, '/target_pose', 10)
        
        # 半球参数定义
        self.center = np.array([0.0, 0.4, 0.4])  # 半球中心
        radius = 0.3  # 半径
        
        # 定义采样步数
        phi_steps = 5
        theta_steps = 8
        
        self.target_positions = []
        
        # 先添加顶点 (phi = 0)
        # 此时sin(0)=0, cos(0)=1，半球最顶点
        x_top = self.center[0] + radius * math.sin(0) * math.cos(0) 
        y_top = self.center[1] + radius * math.sin(0) * math.sin(0)
        z_top = self.center[2] + radius * math.cos(0)
        self.target_positions.append(np.array([x_top, y_top, z_top]))

        # 从i=1开始，避免重复顶点
        for i in range(1, phi_steps+1):
            phi = (math.pi/2)*i/phi_steps
            for j in range(theta_steps):
                theta = 2*math.pi*j/theta_steps
                x = self.center[0] + radius * math.sin(phi) * math.cos(theta)
                y = self.center[1] + radius * math.sin(phi) * math.sin(theta)
                z = self.center[2] + radius * math.cos(phi)
                self.target_positions.append(np.array([x, y, z]))
        self.publish_target_poses()

    def publish_target_poses(self):
        for pos in self.target_positions:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "base_link"
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            
            # 设置位置
            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2])
            
            # 计算Z轴对准球心方向
            direction = self.center - pos
            direction_norm = direction / np.linalg.norm(direction)
            
            # 使用全局up向量
            up = np.array([0.0, 0.0, 1.0])
            # 避免direction与up平行
            if abs(np.dot(direction_norm, up)) > 0.99:
                up = np.array([0.0, 1.0, 0.0])
            
            # 构造正交基：Z轴对准direction_norm
            z_axis = direction_norm
            x_axis = np.cross(up, z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-9:
                up = np.array([0.0, 1.0, 0.0])
                x_axis = np.cross(up, z_axis)
                x_norm = np.linalg.norm(x_axis)
            x_axis = x_axis / x_norm
            y_axis = np.cross(z_axis, x_axis)
            
            # 构建旋转矩阵
            rot_matrix = np.array([
                [x_axis[0], y_axis[0], z_axis[0]],
                [x_axis[1], y_axis[1], z_axis[1]],
                [x_axis[2], y_axis[2], z_axis[2]]
            ])
            
            # 使用scipy的Rotation转换为四元数
            r = R.from_matrix(rot_matrix)
            q = r.as_quat()  # [x, y, z, w]
            
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]

            self.get_logger().info(f"Publishing target pose: pos={pos}, q={q}")
            self.publisher_.publish(pose_msg)
            input("Press Enter to continue...")

def main(args=None):
    rclpy.init(args=args)
    node = HemisphereTargetPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
