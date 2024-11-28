import cv2
import numpy as np
from typing import Tuple

import torch
import torch.nn.functional as F
#from utils.stepfun import sample_np

import numpy as np


def get_intrinsic_matrix(cam):
    # 获取相机的宽度和高度
    width = cam.image_width
    height = cam.image_height

    # 获取相机的视场角
    fov_x = cam.FoVx
    fov_y = cam.FoVy

    # 计算焦距
    fx = width / (2 * np.tan(fov_x / 2))
    fy = height / (2 * np.tan(fov_y / 2))

    # 计算主点坐标
    cx = (width - 1) / 2
    cy = (height - 1) / 2

    # 构建内参矩阵 K
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ]).to('cuda')

    return K

def look_at_matrix(camera_pos, target):
    forward = (target - camera_pos)
    forward /= np.linalg.norm(forward)

    up = np.array([0, 1, 0])
    if np.abs(np.dot(forward, up)) > 0.99:  # 如果接近平行，重新定义up
        up = np.array([1, 0, 0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)  # 归一化

    up = np.cross(forward, right)
    R = np.vstack([right, up, forward]).T
    return R


def compute_target_point(views,radius):
    target_points = []

    for view in views:
        R = view.R  # 旋转矩阵
        T = view.T  # 平移矩阵
        forward_direction = R[:, 2]
        target_point = T + forward_direction*radius*5
        target_points.append(target_point)

    target_point = np.mean(target_points, axis=0)

    return target_point


def compute_radius_range(views):
    T_views = np.array([view.T for view in views])  # 提取所有的平移向量T
    mean_T_views = np.mean(T_views, axis=0)
    distances = np.linalg.norm(T_views - mean_T_views, axis=1)

    return np.mean(distances),np.min(distances),np.max(distances), mean_T_views


def depth_to_new_camera(depth, camera, new_camera):
    # Calculate the transformation from A to B
    def depth_warping_cuda(original_image,depth_map, K, R_A, T_A, R_B, T_B):
        # 将所有矩阵移到GPU并且转置R_A, R_B
        R_A = R_A.transpose(1, 0).to(torch.float32).cuda()
        R_B = R_B.transpose(1, 0).to(torch.float32).cuda()
        depth_map = depth_map.to(torch.float32).cuda()
        scaled_depth = depth_map
        R_AB = torch.matmul(R_B, torch.linalg.inv(R_A))
        T_AB = T_B.to(torch.float32) - torch.matmul(R_AB, T_A.to(torch.float32))
        K = K.to(torch.float32).cuda()
        K_inv = torch.linalg.inv(K)

        height, width = depth_map.shape
        # 在GPU上初始化 warped_depth 和 mask

        warped_depth = torch.full((height, width), float('inf'), dtype=torch.float32, device='cuda')


        # 生成所有像素的索引
        y, x = torch.meshgrid(torch.arange(height, device='cuda', dtype=torch.float32),
                              torch.arange(width, device='cuda', dtype=torch.float32), indexing='ij')
        xy_homog = torch.stack([x, y, torch.ones_like(x)], dim=-1)

        # 将像素坐标转换为归一化相机坐标系下的坐标
        xy_normalized = torch.matmul(K_inv, xy_homog.reshape(-1, 3).T).T

        P_A = scaled_depth.view(-1)[:, None] * xy_normalized

        P_B = torch.matmul(R_AB, P_A.T).T + T_AB.T

        # 投影到目标相机B的图像平面
        xy_b_homog = torch.matmul(K, P_B.T).T
        depth_values = xy_b_homog[:, 2]

        x_b, y_b = (xy_b_homog[:, :2] / xy_b_homog[:, 2, None]).T

        # 对像素坐标进行舍入
        x_b = torch.round(x_b).long()
        y_b = torch.round(y_b).long()

        valid_mask = (x_b >= 0) & (x_b < width) & (y_b >= 0) & (y_b < height)
        unique_indices = y_b[valid_mask] * width + x_b[valid_mask]

        # 找到最小深度
        min_depths = torch.full_like(depth_values, float('inf'), device='cuda')
        min_depths[unique_indices] = torch.min(min_depths[unique_indices],depth_values[valid_mask])

        # 更新有效掩码
        min_depth_mask = torch.zeros_like(valid_mask,dtype=torch.bool,device='cuda')
        min_depth_mask[valid_mask] = (depth_values[valid_mask] == min_depths[unique_indices])

        valid_mask =  (min_depth_mask & valid_mask)
        warped_depth[y_b[valid_mask], x_b[valid_mask]] = depth_values[valid_mask]

        y_a = y.reshape(-1)[valid_mask].long()
        x_a = x.reshape(-1)[valid_mask].long()

        original_image = original_image.permute(1, 2, 0)
        warped_image = torch.zeros_like(original_image, device='cuda')
        warped_image[y_b[valid_mask], x_b[valid_mask]] = original_image[y_a, x_a]

        return warped_depth, warped_image.permute(2, 0, 1)

    src_depth = depth
    src_image = torch.tensor(camera.original_image).cuda()
    src_R = torch.tensor(camera.R).cuda()
    src_T = torch.tensor(camera.T).cuda()
    trg_R = torch.tensor(new_camera.R).cuda()
    trg_T = torch.tensor(new_camera.T).cuda()
    K = get_intrinsic_matrix(camera)

    warped_depth, warped_image  = depth_warping_cuda(src_image,src_depth, K, src_R, src_T, trg_R, trg_T)
    height, width = src_depth.shape
    mask = torch.zeros((height, width), dtype=torch.bool, device='cuda')
    mask[torch.isinf(warped_depth)] = False
    mask[~torch.isinf(warped_depth)] = True
    warped_depth[torch.isinf(warped_depth)] = 255
    return warped_depth,warped_image , mask

def generate_uniform_poses_forview(view, n_frames=20, z_variation=0.1, z_phase=0):
    # 根据views计算景物的中心点
    radius = 0.5
    camera_pos = view.T
    target_point = view.T + view.R[:, 2] * 5
    poses = []

    for i in range(n_frames):
        # 计算圆周上的均匀分布点
        angle = 2 * np.pi * i / n_frames
        x = radius * np.cos(angle) + camera_pos[0]
        y = radius * np.sin(angle) + camera_pos[1]
        z = camera_pos[2] + np.random.uniform(-z_variation, z_variation)  # 随机高度

        camera_pos = np.array([x, y, z])
        R = look_at_matrix(camera_pos, target_point)

        poses.append({'R': R, 'T': camera_pos})

    return poses

def generate_random_poses_forview(view, n_frames=20, z_variation=0.1, z_phase=0):
    # 根据views计算景物的中心点
    radius = 1
    camera_pos = view.T
    target_point = view.T+ view.R[:, 2]*5
    poses = []

    for i in range(n_frames):
        # 随机生成相机在球体上的位置
        theta1 = np.random.uniform(-np.pi/5, -np.pi/8)  # 随机角度
        phi1 = np.random.uniform(-np.pi/5, -np.pi/8)  # 随机高度

        theta2 = np.random.uniform(np.pi/8, np.pi/5)  # 随机角度
        phi2 = np.random.uniform(np.pi / 8, np.pi / 5)  # 随机高度

        theta= np.random.choice([theta1, theta2])
        phi = np.random.choice([phi1, phi2])

        z = np.random.uniform(-0.2, 0.2)  # 随机高度

        x = radius * np.sin(phi) * np.cos(theta) + camera_pos[0]
        y = radius * np.sin(phi) * np.sin(theta) + camera_pos[1]
        z = camera_pos[2] + z

        camera_pos = np.array([x, y, z])
        R = look_at_matrix(camera_pos, target_point)

        poses.append({'R': R, 'T': camera_pos})

    return poses

def generate_random_poses(views, n_frames=10, z_variation=0.1, z_phase=0):
    # 根据views计算景物的中心点
    mean_radius,min_radius,max_radius, mean_camera_pos = compute_radius_range(views)
    target_point = compute_target_point(views,max_radius)

    poses = []

    for i in range(n_frames):
        # 随机生成相机在球体上的位置
        theta = np.random.uniform(0, 2*np.pi)  # 随机角度
        phi = np.random.uniform(0, 2*np.pi)  # 随机高度
        radius =  np.random.uniform(min_radius,mean_radius)  # 随机半径

        x = radius * np.sin(phi) * np.cos(theta) + mean_camera_pos[0]
        y = radius * np.sin(phi) * np.sin(theta) + mean_camera_pos[1]
        z = radius * np.cos(phi) + z_variation * np.sin(2 * np.pi * i / n_frames + z_phase) + mean_camera_pos[2]

        camera_pos = np.array([x, y, z])
        R = look_at_matrix(camera_pos, target_point)

        poses.append({'R': R, 'T': camera_pos})

    return poses