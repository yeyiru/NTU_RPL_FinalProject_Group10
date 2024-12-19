#!/usr/bin/env python3
import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class ExtrinsicTriggeredImagePublisher(Node):
    def __init__(self):
        super().__init__('extrinsic_triggered_image_publisher')

        # 参数：存放图片的文件夹路径
        self.image_folder = '/data/2024Fall/RPL/final/neu-nbv/scripts/neural_rendering/data/dataset/dtu_dataset/rs_dtu_4/DTU/scan1/image'

        # 声明发布者
        self.image_publisher = self.create_publisher(Image, '/camera/sim_image', 10)
        # 設定定時器週期，每隔1秒發布一次外參矩陣
        timer_period = 0.1  
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # 声明订阅者，当收到/extrinsic_matrix后触发回调
        self.extrinsic_subscriber = self.create_subscription(
            PoseStamped, 
            '/target_pose',
            self.extrinsic_callback, 
            10
        )
        self.current_image_index = 0

        self.bridge = CvBridge()

        # 准备读取文件夹中的图像列表
        self.image_files = [f for f in os.listdir(self.image_folder) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.image_files.sort()
        

        self.get_logger().info("Node initialized and ready to receive extrinsic_matrix.")

    def timer_callback(self):
        # 每当收到extrinsic_matrix时，发布一张图像
        if self.current_image_index < len(self.image_files):
            image_path = os.path.join(self.image_folder, self.image_files[self.current_image_index])
            self.get_logger().info(f"Loading image: {image_path}")
            cv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error(f"Failed to read image: {image_path}")
                return

            # 将cv_image转换为ROS的Image消息
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.image_publisher.publish(image_msg)
            self.get_logger().info(f"Published image {self.image_files[self.current_image_index]}")

        else:
            self.get_logger().info("No more images to publish.")
    
    def extrinsic_callback(self, msg):
        # 当收到extrinsic_matrix时，打印消息
        self.get_logger().info(f"Received extrinsic_matrix: {msg}")
        self.current_image_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicTriggeredImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()