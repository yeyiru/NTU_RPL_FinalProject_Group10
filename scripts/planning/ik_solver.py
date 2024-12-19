#ik_solver
import sys
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header, Bool
from builtin_interfaces.msg import Duration

class IKSolver(Node):
    def __init__(self, group_name="ur_manipulator", end_effector_link="tool0"):
        super().__init__('ik_solver')
        self.client = self.create_client(GetPositionIK, 'compute_ik')
        while not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('等待IK服务compute_ik...')

        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.state_publisher = self.create_publisher(Bool, '/ik_solver_state', 10)  # 新增失败信号话题
        self.subscription = self.create_subscription(PoseStamped, '/target_pose', self.compute_ik_and_publish, 10)
        self.group_name = group_name
        self.end_effector_link = end_effector_link

        self.state_signal = True
        # 定时器定期获取Camera位姿
        self.state_timer = self.create_timer(0.05, self.state_callback)

        self.get_logger().info('IKSolver 节点已启动，订阅 /target_pose...')

    def compute_ik_and_publish(self, pose_stamped):
        # 构建逆解请求
        ik_request = PositionIKRequest()
        ik_request.group_name = self.group_name
        # 与test代码保持一致，将avoid_collisions设为True
        ik_request.avoid_collisions = False
        ik_request.timeout.sec = 5   # 设置解析ik超时时间为5秒
        ik_request.pose_stamped = pose_stamped
        ik_request.robot_state.is_diff = True
        ik_request.ik_link_name = self.end_effector_link

        req = GetPositionIK.Request()
        req.ik_request = ik_request

        # 调用逆解服务
        future = self.client.call_async(req)
        future.add_done_callback(self.ik_service_callback)

    def ik_service_callback(self, future):
        try:
            resp = future.result()
            if len(resp.solution.joint_state.name) > 0:
                self.get_logger().info("逆解成功！")
                joint_names = resp.solution.joint_state.name
                joint_positions = resp.solution.joint_state.position

                # 打印关节信息
                for name, pos in zip(joint_names, joint_positions):
                    self.get_logger().info(f"{name}: {math.degrees(pos):.2f}度")

                # 构建 JointTrajectory 消息
                trajectory_msg = JointTrajectory()
                trajectory_msg.header = Header()
                # 将时间戳置为0，让控制器立即执行轨迹
                trajectory_msg.header.stamp.sec = 0
                trajectory_msg.header.stamp.nanosec = 0
                trajectory_msg.joint_names = joint_names

                point = JointTrajectoryPoint()
                point.positions = list(joint_positions)
                point.time_from_start = Duration(sec=2, nanosec=0)
                trajectory_msg.points.append(point)

                # 发布消息
                self.get_logger().info(f"发布的消息内容: {trajectory_msg}")
                self.publisher.publish(trajectory_msg)
                self.state_signal = True
            else:
                self.get_logger().warn("逆解失败，未找到可行解！")
                self.state_signal = False
        except Exception as e:
            self.get_logger().error(f"服务调用时发生异常：{e}")
            self.state_signal = False

    def state_callback(self):
        """发布逆解狀態的信号"""
        state_msg = Bool()
        state_msg.data = self.state_signal
        self.state_publisher.publish(state_msg)
        # self.get_logger().info("已发布逆解状态信号到 /ik_solver_state")

def main(args=None):
    rclpy.init(args=args)
    node = IKSolver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
