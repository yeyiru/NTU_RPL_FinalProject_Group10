import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

import copy
import time
import yaml
import random
import datetime
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
matplotlib.use("TkAgg")

from neural_rendering.evaluation.pretrained_model import PretrainedModel
from neural_rendering.data import get_data
from neural_rendering.utils import parser, util

from dotmap import DotMap

import cv2
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

def setup_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def deg2rad(d):
    return d * np.pi / 180.0

def normalize(v):
    return v / np.linalg.norm(v)

class Tool0PoseNode(Node):
    def __init__(self):
        super().__init__('tool0_pose_node')

        # 创建TF2的Buffer和Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose = [0.0, 0.0, 0.0]
        # 定时器定期获取tool0位姿
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        # 尝试从base_link到tool0查找变换
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            
            # 从变换中获取平移和旋转
            t = trans.transform.translation
            q = trans.transform.rotation
            self.pose = [t.x, t.y, t.z]
        
        except Exception as e:
            self.get_logger().warn('Could not transform: {}'.format(e))
    
    def get_current_pose(self):
        return self.pose

class ExtrinsicPublisher(Node):
    def __init__(self):
        super().__init__('extrinsic_publisher')

        self.publisher_ = self.create_publisher(PoseStamped, 
                                                '/target_pose', 
                                                10)
        self.get_logger().info('ExtrinsicPublisher node initialized.')

    def publish_extrinsic(self, extrinsic):
        # 確保 extrinsic 是 np.ndarray
        extrinsic_array = np.array(extrinsic) if not isinstance(extrinsic, np.ndarray) else extrinsic

        # 從4x4矩陣中取出位置與旋轉矩陣
        translation = extrinsic_array[0:3, 3]
        rotation_mat = extrinsic_array[0:3, 0:3]

        # 使用scipy將旋轉矩陣轉為四元數 (x, y, z, w)
        r = R.from_matrix(rotation_mat)
        quaternion = r.as_quat()  # 回傳順序為 [x, y, z, w]

        # 建立PoseStamped訊息
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp.sec = 0
        pose_stamped.header.stamp.nanosec = 0
        pose_stamped.header.frame_id = 'base_link'

        pose_stamped.pose.position.x = float(translation[0])
        pose_stamped.pose.position.y = float(translation[1])
        pose_stamped.pose.position.z = float(translation[2])

        pose_stamped.pose.orientation.x = float(quaternion[0])
        pose_stamped.pose.orientation.y = float(quaternion[1])
        pose_stamped.pose.orientation.z = float(quaternion[2])
        pose_stamped.pose.orientation.w = float(quaternion[3])

        self.publisher_.publish(pose_stamped)
        self.get_logger().info('Pose published from extrinsic matrix (using scipy).')
        self.get_logger().info(f"x={float(quaternion[0])}, y={float(quaternion[1])}, z={float(quaternion[2])}, w={float(quaternion[3])}")


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        # 訂閱影像話題，需將話題名稱替換為實際的攝影機話題，如"/camera/image"
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.count = 0
        self.latest_raw_image = None
        self.latest_image = None

    def listener_callback(self, msg):
        # 將ROS影像訊息轉成OpenCV影像格式
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_raw_image = cv_image.copy()
        _latest_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        _latest_image = _latest_image.astype(np.float32) / 255.0
        self.latest_image = _latest_image.transpose(2, 0, 1)
        self.count += 1
        # print(f"Received image {self.count}")
    
    def get_latest_raw_image(self):
        return self.latest_raw_image

    def get_latest_image(self):
        return self.latest_image
    
class IkStateSubscriber(Node):
    def __init__(self):
        super().__init__('Ik_state_subscriber')
        # 訂閱影像話題，需將話題名稱替換為實際的攝影機話題，如"/camera/image"
        self.subscription = self.create_subscription(Bool, '/ik_solver_state', self.listener_callback, 10)
        self.count = 0
        self.latest_state = None

    def listener_callback(self, msg):
        # 將ROS影像訊息轉成OpenCV影像格式
        self.latest_state = msg.data
        # print(f"Received image {self.count}")
    
    def get_latest_state(self):
        return self.latest_state

class TM5NBVPlanner:
    def __init__(self):
        with open('./scripts/planning/config/tm5_planner.yaml', 'r') as f:
            config = yaml.safe_load(f)
        with open(f"{config['model_path']}/training_setup.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)

        gpu_id = [config['cuda']]
        self.device = util.get_cuda(gpu_id[0])
        
        checkpoint_path = os.path.join(config['model_path'], "checkpoints", "best.ckpt")
        assert os.path.exists(checkpoint_path), "checkpoint does not exist"
        ckpt_file = torch.load(checkpoint_path)
        
        self.model = PretrainedModel(model_cfg["model"], ckpt_file, self.device, gpu_id)
        
        # 定义半球的中心
        self.center = np.array(config['center'], dtype=np.float64)
        # 定义目标点(拍摄对准中心)
        self.target = self.center
        self.radius = config['radius']
        # 取樣水平角（方位角）
        self.phi_steps = config['phi_steps']
        # 取樣垂直角（极角）
        self.theta_steps = config['theta_steps']
        self.up = np.array(config['up'], dtype=np.float64)

        # 相機內參
        self.focal = torch.tensor(config['focal'], dtype=torch.float32).to(self.device)
        self.c = torch.tensor(config['c'], dtype=torch.float32).to(self.device)

        self.budget = config['budget']

        self.z_near = config['z_near']
        self.z_far = config['z_far']

        # 判断是否显示结果
        self.show_result = True

        # 建立保存實驗結果的文件夾
        self.experiment_path = os.path.join('./scripts/experiments/TM5_test/', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.experiment_path, exist_ok=True)
    
    def sample_point(self):
        # 用于存储 (camera_pos, R) 的列表
        results = []
        target_positions = []

        x_top = self.center[0] + self.radius * np.sin(0) * np.cos(0) 
        y_top = self.center[1] + self.radius * np.sin(0) * np.sin(0)
        z_top = self.center[2] + self.radius * np.cos(0)
        target_positions.append(np.array([x_top, y_top, z_top]))

        # 从i=1开始，避免重复顶点
        for i in range(1, self.phi_steps+1):
            phi = (np.pi/2)*i/self.phi_steps
            for j in range(self.theta_steps):
                theta = 2*np.pi*j / self.theta_steps
                x = self.center[0] + self.radius * np.sin(phi) * np.cos(theta)
                y = self.center[1] + self.radius * np.sin(phi) * np.sin(theta)
                z = self.center[2] + self.radius * np.cos(phi)
                target_positions.append(np.array([x, y, z]))
        
        for pos in target_positions:
            direction = self.center - pos
            direction_norm = direction / np.linalg.norm(direction)
            up = self.up
            if abs(np.dot(direction_norm, up)) > 0.99:
                up = np.array([0.0, 1.0, 0.0])
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
            results.append((pos, rot_matrix))
        return results

    def tool0_to_camera(self):
        self.camera_poses = []
        adjust_matrix = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0,  1]
        ])
        ro_z_90 = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        for i in range(len(self.tool0_poses)):
            pos, R = self.tool0_poses[i]
            _pos = pos + np.array([0.0, 0.0, 0.01])
            _R = R @ adjust_matrix
            _R = _R @ ro_z_90 @ ro_z_90 
            self.camera_poses.append([_pos, _R])
        self.tool0_camera_poses = list(zip(self.tool0_poses, self.camera_poses))
        return self.tool0_camera_poses

    def plot_cameras_in_3D(self):
        
        fig = plt.figure(figsize=(20,16))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制相机位置散点
        for (tool0_poses, camera_poses) in self.tool0_camera_poses:
            pos, R = camera_poses
            ax.scatter(pos[0], pos[1], pos[2], c='r', marker='o')

            # 绘制相机坐标轴
            # R的行向量：r, u, -f分别对应x,y,z轴方向
            cam_x = R[:, 0]
            cam_y = R[:, 1]
            cam_z = R[:, 2]

            scale = 0.2  # 坐标轴长度缩放因子
           
            # 绘制x轴(红色)
            ax.quiver(pos[0], pos[1], pos[2],
                      cam_x[0]*scale, cam_x[1]*scale, cam_x[2]*scale,
                      color='r', linewidth=2)
            # 绘制y轴(绿色)
            ax.quiver(pos[0], pos[1], pos[2],
                    cam_y[0]*scale, cam_y[1]*scale, cam_y[2]*scale,
                    color='g', linewidth=2)
            # 绘制z轴(蓝色)
            ax.quiver(pos[0], pos[1], pos[2],
                    cam_z[0]*scale, cam_z[1]*scale, cam_z[2]*scale,
                    color='b', linewidth=2)

        # 绘制半球点云做参考
        # theta从0到90度, phi从0到360度
        phi_lin = np.linspace(0, 2*np.pi, 36)
        theta_lin = np.linspace(0, np.pi/2, 10)
        phi_grid, theta_grid = np.meshgrid(phi_lin, theta_lin)

        X = self.center[0] + self.radius * np.sin(theta_grid)*np.cos(phi_grid)
        Y = self.center[1] + self.radius * np.sin(theta_grid)*np.sin(phi_grid)
        Z = self.center[2] + self.radius * np.cos(theta_grid)

        ax.plot_surface(X, Y, Z, alpha=0.1, color='gray', edgecolor='none')

        # 在半球圆心处添加一个小立方块
        cube_half = 0.1 # 半边长
        cx, cy, cz = self.center
        cz += cube_half

        # 定义立方体的8个顶点
        vertices = np.array([
            [cx - cube_half, cy - cube_half, cz - cube_half],
            [cx + cube_half, cy - cube_half, cz - cube_half],
            [cx + cube_half, cy + cube_half, cz - cube_half],
            [cx - cube_half, cy + cube_half, cz - cube_half],
            [cx - cube_half, cy - cube_half, cz + cube_half],
            [cx + cube_half, cy - cube_half, cz + cube_half],
            [cx + cube_half, cy + cube_half, cz + cube_half],
            [cx - cube_half, cy + cube_half, cz + cube_half]
        ])

        # 立方体的6个面
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]], # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]], # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]], # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]], # 后面
            [vertices[1], vertices[2], vertices[6], vertices[5]], # 右面
            [vertices[0], vertices[3], vertices[7], vertices[4]]  # 左面
        ]
        cube = Poly3DCollection(faces, linewidths=1, edgecolors='k', alpha=0.7)
        cube.set_facecolor('cyan')
        ax.add_collection3d(cube)

        # 在 0,0,0 处添加一个细长的棒表示手臂的位置
        arm_height = 1.0  # 手臂高度
        arm_width = 0.05   # 手臂宽度
        arm_depth = 0.05  # 手臂深度

        # 中心点位置
        cx, cy, cz = (0, 0, 0)
        cz += arm_height / 2  # 长棒向上的一半长度

        # 定义细长棒的8个顶点
        vertices = np.array([
            [cx - arm_width / 2, cy - arm_depth / 2, cz - arm_height / 2],
            [cx + arm_width / 2, cy - arm_depth / 2, cz - arm_height / 2],
            [cx + arm_width / 2, cy + arm_depth / 2, cz - arm_height / 2],
            [cx - arm_width / 2, cy + arm_depth / 2, cz - arm_height / 2],
            [cx - arm_width / 2, cy - arm_depth / 2, cz + arm_height / 2],
            [cx + arm_width / 2, cy - arm_depth / 2, cz + arm_height / 2],
            [cx + arm_width / 2, cy + arm_depth / 2, cz + arm_height / 2],
            [cx - arm_width / 2, cy + arm_depth / 2, cz + arm_height / 2]
        ])

        # 定义细长棒的6个面
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # 左面
        ]

        # 绘制细长棒
        arm = Poly3DCollection(faces, linewidths=1, edgecolors='k', alpha=0.7)
        arm.set_facecolor('cyan')
        # ax.add_collection3d(arm)

        # 设置坐标轴范围
        ax.set_xlim(self.center[0]-0.8, self.center[0]+0.8)
        ax.set_ylim(-0.5, 1.2)
        ax.set_zlim(0.0, 1.2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Camera Positions and Orientations")
        plt.tight_layout()
        plt.show()
        # plt.savefig("camera_poses.png")
    
    def initial_poses(self):
        # 从tool0_poses中随机选择二个作为初始参考视角
        candidate_poses = []
        camera_candidate_poses = []
        _ref_poses = copy.deepcopy(self.tool0_camera_poses)
        random.shuffle(_ref_poses)

        for ref_pose, camera_pose in _ref_poses:
            # 建立4x4外參矩陣
            extrinsic_matrix = np.eye(4)  # 先建立一個4x4單位矩陣
            extrinsic_matrix[:3, :3] = ref_pose[1]  # 前3x3區域放入旋轉矩陣
            extrinsic_matrix[:3, 3] = ref_pose[0]   # 第四欄前3列放入平移向量
            candidate_poses.append(extrinsic_matrix)

            # 对camera pose也做同样的处理
            camera_extrinsic_matrix = np.eye(4)  # 先建立一個4x4單位矩陣
            camera_extrinsic_matrix[:3, :3] = camera_pose[1]  # 前3x3區域放入旋轉矩陣
            camera_extrinsic_matrix[:3, 3] = camera_pose[0]   # 第四欄前3列放入平移向量
            camera_candidate_poses.append(extrinsic_matrix)


        candidate_poses = torch.tensor(candidate_poses, dtype=torch.float32).to(self.device)
        camera_candidate_poses = torch.tensor(camera_candidate_poses, dtype=torch.float32).to(self.device)
        return candidate_poses, camera_candidate_poses

    def get_nbv_ref_pose(self):
        _, _, H, W = self.ref_images.shape
        # 把资料丢到model里面

        self.model.network.encode(
            self.ref_images.unsqueeze(0),
            self.ref_camera_poses.unsqueeze(0),
            self.focal.unsqueeze(0),
            self.c.unsqueeze(0),
        )
        reward_list = []
        for target_pose in self.camera_candidate_poses:
            target_rays = util.gen_rays(
                target_pose.unsqueeze(0),
                W, H, self.focal, self.z_near, self.z_far, self.c
            )
            target_rays = target_rays.reshape(1, H * W, -1)
            t_model = self.model.renderer_par(target_rays)
            predict = DotMap(t_model)
            uncertainty = predict["uncertainty"][0]
            # 根据不确定性计算reward
            reward = torch.sum(uncertainty**2).cpu().numpy()
            reward_list.append(reward)
        nbv_index = np.argsort(-np.array(reward_list))
        new_ref_poses = self.candidate_poses[nbv_index]
        new_ref_camera_poses = self.camera_candidate_poses[nbv_index]
        self.ref_poses = torch.cat((self.ref_poses, new_ref_poses), dim=0)
        self.ref_camera_poses = torch.cat((self.ref_camera_poses, new_ref_camera_poses), dim=0)
        return self.ref_poses, self.ref_camera_poses
    
    def render_image(self):
        _, _, H, W = self.ref_images.shape
        self.model.network.encode(
            self.final_images.unsqueeze(0),
            self.final_camera_poses.unsqueeze(0),
            self.focal.unsqueeze(0),
            self.c.unsqueeze(0),
        )
        cnt = 0
        for target_pose in self.all_camera_candidate_poses:
            target_rays = util.gen_rays(
                target_pose.unsqueeze(0), 
                W, H, self.focal, self.z_near, self.z_far, self.c
            )
            target_rays = target_rays.reshape(1, H * W, -1)
            predict = DotMap(self.model.renderer_par(target_rays))
            rgb = predict.rgb[0].cpu().reshape(H, W, 3).numpy() * 255
            cv2.imwrite(f"{self.experiment_path}/{cnt}.png", rgb)
            cnt += 1
        return

    def main(self):
        rclpy.init(args=None)
        extrinsic_publisher = ExtrinsicPublisher()
        image_subscriber = ImageSubscriber()
        state_subscriber = IkStateSubscriber()
        current_pose_subscriber = Tool0PoseNode()
        executor = MultiThreadedExecutor()
        executor.add_node(image_subscriber)
        executor.add_node(state_subscriber)
        executor.add_node(current_pose_subscriber)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        with torch.no_grad():
            self.tool0_poses = self.sample_point()
            self.tool0_camera_poses = self.tool0_to_camera()
            if self.show_result:
                self.plot_cameras_in_3D()

            
            # self.candidate_poses，将所有的tool0_poses转换成4x4的矩阵
            self.candidate_poses, self.camera_candidate_poses = self.initial_poses()
            self.all_candidate_poses = copy.deepcopy(self.candidate_poses)
            self.all_camera_candidate_poses = copy.deepcopy(self.camera_candidate_poses)
            
            # 初始化一个list来存储ref_images
            self.ref_images = []
            # 初始化一个2x4x4的矩阵来存储候选的ref_poses
            self.ref_poses = torch.zeros(2, 4, 4, dtype=torch.float32).to(self.device)
            self.ref_camera_poses = torch.zeros(2, 4, 4, dtype=torch.float32).to(self.device)
            
            # 这个for loop获得2个初始的位置
            cnt = 0
            for i in range(len(self.candidate_poses)):
                ref_pose = self.candidate_poses[i]
                _img = None
                extrinsic_publisher.publish_extrinsic(ref_pose.cpu().numpy())
                # 延迟5s等待逆解完成
                time.sleep(8)
                t = state_subscriber.get_latest_state()
                if state_subscriber.get_latest_state():
                    # 逆解成功，把这个姿态加入到self.ref_poses中
                    self.ref_poses[cnt, ...] = ref_pose
                    self.ref_camera_poses[cnt, ...] = self.camera_candidate_poses[i]
                    
                    # 等待机器人移动到位
                    p1 = np.array(current_pose_subscriber.get_current_pose())
                    p2 = ref_pose[:3, 3].cpu().numpy()
                    while np.linalg.norm(p1-p2) > 0.01:
                        time.sleep(0.5)
                        print('Waiting for the robot to move')

                    print('Move done, Ready to get image')
                    while _img is None:
                        _img = image_subscriber.get_latest_image()
                    raw = _img.transpose(1, 2, 0)
                    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) * 255
                    cv2.imwrite(f"{self.experiment_path}/raw_{cnt}.png", raw)
                    
                    print('Got image')
                    self.ref_images.append(_img)
                    cnt += 1
                
                if len(self.ref_images) == 2:
                    # 从self.candidate_poses中删除已经选择或者ik失败的pose
                    self.candidate_poses = self.candidate_poses[i+1:]
                    self.camera_candidate_poses = self.camera_candidate_poses[i+1:]
                    break
            self.ref_images = torch.tensor(self.ref_images, dtype=torch.float32).to(self.device)

            print('Calculating another poses...')
            s0 = time.time()
            self.ref_poses, self.ref_camera_poses = self.get_nbv_ref_pose()
            s1 = time.time()
            print(f'Calculate done, Spend time {s1-s0}, We get {self.ref_poses.shape[0]} ref poses')
            cnt = 2
            self.final_poses = torch.zeros(self.budget, 4, 4, dtype=torch.float32).to(self.device)
            self.final_camera_poses = torch.zeros(self.budget, 4, 4, dtype=torch.float32).to(self.device)
            self.final_poses[:2, ...] = self.ref_poses[:2, ...]
            self.final_camera_poses[:2, ...] = self.ref_camera_poses[:2, ...]

            self.final_images = torch.zeros(self.budget, 
                                            self.ref_images.shape[1], 
                                            self.ref_images.shape[2], 
                                            self.ref_images.shape[3], dtype=torch.float32).to(self.device)
            self.final_images[:2, ...] = self.ref_images
            # 获得剩下的几个点的图像
            for i in range(len(self.ref_poses[2:, ...])):
                ref_pose = self.ref_poses[i+2, ...]
                ref_camera_pose = self.ref_camera_poses[i+2, ...]
                _img = None
                extrinsic_publisher.publish_extrinsic(ref_pose.cpu().numpy())
                # 延迟5s等待逆解完成
                time.sleep(8)
                if state_subscriber.get_latest_state():
                    # 逆解成功，把这个姿态加入到self.final_poses中
                    self.final_poses[cnt, ...] = ref_pose
                    self.final_camera_poses[cnt, ...] = ref_camera_pose
                    # 等待机器人移动到位
                    p1 = np.array(current_pose_subscriber.get_current_pose())
                    p2 = ref_pose[:3, 3].cpu().numpy()
                    t = p1 - p2
                    while np.linalg.norm(p1-p2) > 0.01:
                        time.sleep(0.5)
                        print('Waiting for the robot to move...')

                    print('Move done, Ready to get image')
                    while _img is None:
                        _img = image_subscriber.get_latest_image()
                    raw = _img.transpose(1, 2, 0)
                    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) * 255
                    cv2.imwrite(f"{self.experiment_path}/raw_{cnt}.png", raw)

                    print('Got image')
                    self.ref_images = torch.cat((self.ref_images, 
                                                torch.tensor(_img, dtype=torch.float32, device=self.device).unsqueeze(0)), 
                                                dim=0)
                    print(f'Got image No.{cnt}')
                    cnt += 1
                if cnt == self.budget:
                    break
                # self.ref_images = torch.tensor(self.ref_images, dtype=torch.float32).to(self.device)
            print('All images are collected')
            
            # self.render_image()
        return

if __name__ == "__main__":
    setup_random_seed(1)

    tm5_nbv_planner = TM5NBVPlanner()
    tm5_nbv_planner.main()
    # plot_cameras_in_3D(camera_poses, center, radius)
    
