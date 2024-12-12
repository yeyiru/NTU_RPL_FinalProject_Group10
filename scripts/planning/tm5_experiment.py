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
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge

def setup_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def deg2rad(d):
    return d * np.pi / 180.0

# class ExtrinsicPublisher(Node):
#     def __init__(self):
#         super().__init__('extrinsic_publisher')
#         self.publisher_ = self.create_publisher(Float32MultiArray, '/extrinsic_matrix', 10)

#         # 設定定時器週期，每隔1秒發布一次外參矩陣
#         timer_period = 1.0  
#         self.timer = self.create_timer(timer_period, self.timer_callback)

#         # 範例外參矩陣(此處以單位矩陣代表，實務上應該用你的實際4x4外參矩陣)
#         self.extrinsic = np.eye(4, dtype=np.float32)

#     def timer_callback(self):
#         msg = Float32MultiArray()
#         # 設定Matrix的維度資訊
#         msg.layout.dim.append(MultiArrayDimension())
#         msg.layout.dim.append(MultiArrayDimension())
#         msg.layout.dim[0].label = "height"
#         msg.layout.dim[0].size = 4
#         msg.layout.dim[0].stride = 16
#         msg.layout.dim[1].label = "width"
#         msg.layout.dim[1].size = 4
#         msg.layout.dim[1].stride = 4

#         # 將4x4矩陣攤平存入Float32MultiArray
#         msg.data = self.extrinsic.flatten().tolist()

#         self.publisher_.publish(msg)
#         self.get_logger().info('Publishing extrinsic matrix')

class ExtrinsicPublisher(Node):
    def __init__(self):
        super().__init__('extrinsic_publisher')

        self.publisher_ = self.create_publisher(Float32MultiArray, '/extrinsic_matrix', 10)
        self.get_logger().info('ExtrinsicPublisher node initialized.')

    def publish_extrinsic(self, extrinsic):
        msg = Float32MultiArray()
        # 設定矩陣維度資訊
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "height"
        msg.layout.dim[0].size = 4
        msg.layout.dim[0].stride = 16
        msg.layout.dim[1].label = "width"
        msg.layout.dim[1].size = 4
        msg.layout.dim[1].stride = 4
        # 將4x4矩陣攤平成list
        msg.data = extrinsic.flatten().tolist()
        self.publisher_.publish(msg)

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        # 訂閱影像話題，需將話題名稱替換為實際的攝影機話題，如"/camera/image"
        self.subscription = self.create_subscription(Image, '/camera/sim_image', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.count = 0
        self.latest_image = None

    def listener_callback(self, msg):
        # 將ROS影像訊息轉成OpenCV影像格式
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.latest_image = self.latest_image.astype(np.float32) / 255.0
        self.latest_image = self.latest_image.transpose(2, 0, 1)
        self.count += 1
        # print(f"Received image {self.count}")
    
    def get_latest_image(self):
        return self.latest_image

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
        self.phi_list = config['phi_list']
        # 取樣垂直角（极角）
        self.theta_list = config['theta_list']
        self.up = np.array(config['up'], dtype=np.float64)

        # 相機內參
        self.focal = torch.tensor(config['focal'], dtype=torch.float32).to(self.device)
        self.c = torch.tensor(config['c'], dtype=torch.float32).to(self.device)

        self.budget = config['budget']

        self.z_near = config['z_near']
        self.z_far = config['z_far']

        # 判断是否显示结果
        self.show_result = False

        # 建立保存實驗結果的文件夾
        self.experiment_path = os.path.join('./scripts/experiments/TM5_test/', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.experiment_path, exist_ok=True)
    
    def sample_point(self):
        camera_poses = []
        for theta_deg in self.theta_list:
            for phi_deg in self.phi_list:
                theta = deg2rad(theta_deg)
                phi = deg2rad(phi_deg)

                # 球面坐标转直角坐标（相机所在点）
                x = self.center[0] + self.radius * np.sin(theta) * np.cos(phi)
                y = self.center[1] + self.radius * np.sin(theta) * np.sin(phi)
                z = self.center[2] + self.radius * np.cos(theta)
                camera_pos = np.array([x, y, z], dtype=np.float64)

                # 构造旋转矩阵
                # 前向向量f (从相机指向目标)
                f = self.target - camera_pos
                f = f / np.linalg.norm(f)

                # 右向量r
                r = np.cross(f, self.up)
                r_norm = np.linalg.norm(r)
                # 如果f与up平行可能导致r=0，需要处理这个情况
                if r_norm < 1e-8:
                    # 如果f与up平行，选择一个不同的上向量进行再次计算
                    # 此处简单处理，如果f接近(0,0,1),则使用(0,1,0)为新的up
                    # 或者(1,0,0)等任意与f不平行的向量
                    alt_up = np.array([0,1,0], dtype=np.float64)
                    r = np.cross(f, alt_up)
                    r_norm = np.linalg.norm(r)
                r = r / r_norm

                # 上向量u
                u = np.cross(r, f)
                u = u / np.linalg.norm(u)

                # 构造相机旋转矩阵 R (从世界坐标到相机坐标)
                # camera的前向(z_cam)为 -f，因此第三行是 -f
                R = np.array([
                    [r[0], r[1], r[2]],
                    [u[0], u[1], u[2]],
                    [-f[0], -f[1], -f[2]]
                ])

                # 构造外参矩阵 (4x4)
                # p_c = R*(p_w - C)，故外参为 [R | -R*C; 0 0 0 1]
                t = -R @ camera_pos
                M_ext = np.eye(4)
                M_ext[:3,:3] = R
                M_ext[:3, 3] = t

                # print("---------------------------------------------------")
                # print(f"Theta: {theta_deg} deg, Phi: {phi_deg} deg")
                # print("Camera Position (XYZ):", camera_pos)
                # print("Camera Rotation Matrix (3x3):\n", R)
                # print("Camera Extrinsic Matrix (4x4):\n", M_ext)
                camera_poses.append((camera_pos, R))
        return camera_poses

    def plot_cameras_in_3D(self):
        
        fig = plt.figure(figsize=(20,16))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制相机位置散点
        for (pos, R) in self.camera_poses:
            ax.scatter(pos[0], pos[1], pos[2], c='r', marker='o')

            # 绘制相机坐标轴
            # R的行向量：r, u, -f分别对应x,y,z轴方向
            cam_x = R[0,:]
            cam_y = R[1,:]
            cam_z = R[2,:]

            scale = 0.2  # 坐标轴长度缩放因子
            # 绘制x轴(红色)
            ax.quiver(pos[0], pos[1], pos[2],
                    cam_x[0]*scale, cam_x[1]*scale, cam_x[2]*scale,
                    color='r', linewidth=2)
            # 绘制y轴(绿色)
            ax.quiver(pos[0], pos[1], pos[2],
                    -cam_y[0]*scale, -cam_y[1]*scale, -cam_y[2]*scale,
                    color='g', linewidth=2)
            # 绘制z轴(蓝色)
            ax.quiver(pos[0], pos[1], pos[2],
                    -cam_z[0]*scale, -cam_z[1]*scale, -cam_z[2]*scale,
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

        # 设置坐标轴范围
        ax.set_xlim(self.center[0]-1.5, self.center[0]+1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0.0, 2.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Camera Positions and Orientations")
        plt.tight_layout()
        plt.show()
        # plt.savefig("camera_poses.png")
    
    def get_initial_ref_poses(self):
        # 从camera_poses中随机选择二个作为初始参考视角
        ref_poses = []
        candidate_poses = []
        _ref_poses = copy.deepcopy(self.camera_poses)
        random.shuffle(_ref_poses)
        cnt = 0
        for ref_pose in _ref_poses:
            # 建立4x4外參矩陣
            extrinsic_matrix = np.eye(4)  # 先建立一個4x4單位矩陣
            extrinsic_matrix[:3, :3] = ref_pose[1]  # 前3x3區域放入旋轉矩陣
            extrinsic_matrix[:3, 3] = ref_pose[0]   # 第四欄前3列放入平移向量
            if cnt < 2:
                ref_poses.append(extrinsic_matrix)
            else:
                candidate_poses.append(extrinsic_matrix)
            cnt += 1
        ref_poses = torch.tensor(ref_poses, dtype=torch.float32).to(self.device)
        candidate_poses = torch.tensor(candidate_poses, dtype=torch.float32).to(self.device)
        return ref_poses, candidate_poses

    def get_nbv_ref_pose(self):
        _, _, H, W = self.ref_images.shape
        # 把资料丢到model里面

        self.model.network.encode(
            self.ref_images.unsqueeze(0),
            self.ref_poses.unsqueeze(0),
            self.focal.unsqueeze(0),
            self.c.unsqueeze(0),
        )
        for i in range(self.budget - 2):
            reward_list = []
            for target_pose in self.candidate_poses:
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
            nbv_index = np.argmax(reward_list)
            new_ref_poses = self.candidate_poses[nbv_index]
            self.ref_poses = torch.cat((self.ref_poses, new_ref_poses.unsqueeze(0)), dim=0)
        return self.ref_poses, list(set(self.candidate_poses) - set(self.ref_poses))
    
    def render_image(self):
        _, _, H, W = self.ref_images.shape
        self.model.network.encode(
            self.ref_images.unsqueeze(0),
            self.ref_poses.unsqueeze(0),
            self.focal.unsqueeze(0),
            self.c.unsqueeze(0),
        )
        cnt = 0
        for target_pose in self.remain_poses:
            target_rays = util.gen_rays(
                target_pose.unsqueeze(0), 
                W, H, self.focal, self.z_near, self.z_far, self.c
            )
            target_rays = target_rays.reshape(1, H * W, -1)
            predict = DotMap(self.model.renderer_par(target_rays))
            rgb = predict.rgb[0].cpu().reshape(H, W, 3).numpy() * 255
            cv2.imwrite(f"{self.experiment_path}/{cnt}.png", rgb)
            cnt+=1
        return

    def main(self):
        rclpy.init(args=None)
        extrinsic_publisher = ExtrinsicPublisher()
        image_subscriber = ImageSubscriber()
        executor = MultiThreadedExecutor()
        executor.add_node(image_subscriber)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        with torch.no_grad():
            self.camera_poses = self.sample_point()
            if self.show_result:
                self.plot_cameras_in_3D()
            self.ref_poses, self.candidate_poses = self.get_initial_ref_poses()
            self.ref_images = []
            for ref_pose in self.ref_poses:
                _img = None
                extrinsic_publisher.publish_extrinsic(ref_pose)
                # 延迟5s等待末端移动到位
                time.sleep(5)
                print('Ready to get image')
                while _img is None:
                    _img = image_subscriber.get_latest_image()
                print('Got image')
                self.ref_images.append(_img)
            self.ref_images = torch.tensor(self.ref_images, dtype=torch.float32).to(self.device)
            self.ref_poses, self.remain_poses = self.get_nbv_ref_pose()
            print(self.ref_poses.shape)
            cnt = 2
            # 获得剩下的几个点的图像
            for ref_pose in self.ref_poses[2:]:
                _img = None
                extrinsic_publisher.publish_extrinsic(ref_pose)
                time.sleep(5)
                while _img is None:
                    _img = image_subscriber.get_latest_image()
                self.ref_images = torch.cat((self.ref_images, 
                                             torch.tensor(_img, dtype=torch.float32, device=self.device).unsqueeze(0)), 
                                             dim=0)
                print(f'Got image No.{cnt}')
                cnt += 1
                # self.ref_images = torch.tensor(self.ref_images, dtype=torch.float32).to(self.device)
            print('All images are collected')
            self.render_image()
        return

if __name__ == "__main__":
    setup_random_seed(10)
    # camera_poses, center, radius = planner()
    # extrinsic_pub = ExtrinsicPublisher()
    # image_sub = ImageSubscriber()
    # executor = MultiThreadedExecutor(num_threads=2)
    # executor.add_node(extrinsic_pub)
    # executor.add_node(image_sub)

    # try:
    #     executor.spin()
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     executor.shutdown()
    #     extrinsic_pub.destroy_node()
    #     image_sub.destroy_node()
    #     rclpy.shutdown()

        

    tm5_nbv_planner = TM5NBVPlanner()
    tm5_nbv_planner.main()
    # plot_cameras_in_3D(camera_poses, center, radius)
    
