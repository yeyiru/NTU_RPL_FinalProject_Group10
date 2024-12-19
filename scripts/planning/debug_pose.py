#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import numpy as np
from scipy.spatial.transform import Rotation as R

class Tool0PoseNode(Node):
    def __init__(self):
        super().__init__('camera_pose_node')

        # 创建TF2的Buffer和Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 定时器定期获取相机位姿
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # 尝试从base_link到tool0查找变换
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
            
            # 从变换中获取平移和旋转
            t = trans.transform.translation
            q = trans.transform.rotation
            
            # 使用scipy从四元数获取旋转矩阵
            # 四元数格式为 [x, y, z, w]
            quat = [q.x, q.y, q.z, q.w]
            rotation = R.from_quat(quat)
            R_mat = rotation.as_matrix()

            # 构造4x4外参矩阵
            M_ext = np.eye(4)
            M_ext[0:3, 0:3] = R_mat
            M_ext[0:3, 3] = [t.x, t.y, t.z]
            
            matrix_str = np.array2string(M_ext, formatter={'float_kind':lambda x: f"{x:.5f}"})
            self.get_logger().info('Camera Pose (4x4):\n{}'.format(matrix_str))

            # 利用四元数构造旋转
            r = R.from_quat(quat)

            # 将旋转转换为欧拉角
            # 这里的 'zyx' 表示先绕 X 轴的旋转(roll)，再绕 Y 轴(pitch)，最后绕 Z 轴(yaw)
            # 具体的旋转顺序需要根据你的坐标定义来选择，比如:
            # 'zyx' 通常对应 roll (绕X), pitch (绕Y), yaw (绕Z)
            euler_angles = r.as_euler('zyx', degrees=True)

            roll, pitch, yaw = euler_angles
            print("Roll: ", roll, "°")
            print("Pitch:", pitch, "°")
            print("Yaw:  ", yaw, "°")
            
        except Exception as e:
            self.get_logger().warn('Could not transform: {}'.format(e))

def main(args=None):
    rclpy.init(args=args)
    node = Tool0PoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
