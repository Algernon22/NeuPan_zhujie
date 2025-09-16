'''
neupan file is the main class for the NeuPan algorithm. It wraps the PAN class and provides a more user-friendly interface.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
'''

import yaml
import torch
from neupan.robot import robot
from neupan.blocks import InitialPath, PAN
from neupan import configuration                              # 全局配置 使用configuration.  使用其中全局变量
from neupan.util import time_it, file_check, get_transform
import numpy as np
from neupan.configuration import np_to_tensor, tensor_to_np
from math import cos, sin


class neupan(torch.nn.Module):

    """
    neupan 类是 NeuPAN 算法的“统一入口”：
    把传感器点云与当前状态转换为安全、可执行的控制动作，内部以 DUNE+NRMP 的近端交替最小化实现端到端、实时、带约束的局部路径规划与控制。
    Args:
        receding: int, the number of steps in the receding horizon.
        step_time: float, the time step in the MPC framework.
        ref_speed: float, the reference speed of the robot.
        device: str, the device to run the algorithm on. 'cpu' or 'cuda'.
        robot_kwargs: dict, the keyword arguments for the robot class.
        ipath_kwargs: dict, the keyword arguments for the initial path class.
        pan_kwargs: dict, the keyword arguments for the PAN class.
        adjust_kwargs: dict, the keyword arguments for the adjust class
        train_kwargs: dict, the keyword arguments for the train class
        time_print: bool, whether to print the forward time of the algorithm.
        collision_threshold: float, the threshold for the collision detection. If collision, the algorithm will stop.
    """

    # 创建类实例时自动执行__init__方法
    def __init__(
        self,
        receding: int = 10,           # 时域长度
        step_time: float = 0.1,       # 时间步长
        ref_speed: float = 4.0,       # 参考速度  
        device: str = "cpu",
        robot_kwargs: dict = None,
        ipath_kwargs: dict = None,
        pan_kwargs: dict = None,      # 近端交替最小化网络参数
        adjust_kwargs: dict = None,
        train_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super(neupan, self).__init__()

        # mpc parameters 
        self.T = receding
        self.dt = step_time
        self.ref_speed = ref_speed

        configuration.device = torch.device(device)
        configuration.time_print = kwargs.get("time_print", False)

        # 碰撞阈值（停止条件）
        # 从 kwargs 字典中获取一个名为 "collision_threshold" 的可选参数，并为 self.collision_threshold 属性赋值。如果该参数没有提供，则使用默认值 0.1
        self.collision_threshold = kwargs.get("collision_threshold", 0.1)

        # 初始化MPC窗口速度序列
        self.cur_vel_array = np.zeros((2, self.T))
        self.robot = robot(receding, step_time, **robot_kwargs)

        self.ipath = InitialPath(
            receding, step_time, ref_speed, self.robot, **ipath_kwargs
        )
            
        pan_kwargs["adjust_kwargs"] = adjust_kwargs
        pan_kwargs["train_kwargs"] = train_kwargs
        self.dune_train_kwargs = train_kwargs

        # 核心近端交替最小化网络（DUNE筛点+NRMP优化）
        self.pan = PAN(receding, step_time, self.robot, **pan_kwargs)

        self.info = {"stop": False, "arrive": False, "collision": False}

    # 类方法可以通过类本身调用，而不需要创建类的实例
    @classmethod
    def init_from_yaml(cls, yaml_file, **kwargs):
        abs_path = file_check(yaml_file)

        with open(abs_path, "r") as f:    # 打开yaml文件
            config = yaml.safe_load(f)    # 读取并解析yaml内容为字典
            config.update(kwargs)         # 用额外传入的参数覆盖/补充配置

        # 将yaml中的各部分参数分别弹出，重命名为kwargs形式，方便后续传递
        config["robot_kwargs"] = config.pop("robot", dict())
        config["ipath_kwargs"] = config.pop("ipath", dict())
        config["pan_kwargs"] = config.pop("pan", dict())
        config["adjust_kwargs"] = config.pop("adjust", dict())
        config["train_kwargs"] = config.pop("train", dict())

        return cls(**config)  # 用整理好的参数字典初始化neupan对象并返回

    # 装饰器 用于测量函数执行时间
    @time_it("neupan forward")
    def forward(self, state, points, velocities=None):
        """
        state: current state of the robot, matrix (3, 1), x, y, theta
        points: current input obstacle point positions, matrix (2, N), N is the number of obstacle points.
        velocities: current velocity of each obstacle point, matrix (2, N), N is the number of obstacle points. vx, vy
        """

        assert state.shape[0] >= 3

        if self.ipath.check_arrive(state):
            self.info["arrive"] = True
            return np.zeros((2, 1)), self.info

        nom_input_np = self.ipath.generate_nom_ref_state(
            state, self.cur_vel_array, self.ref_speed
        )

        # NumPy → Tensor
        nom_input_tensor = [np_to_tensor(n) for n in nom_input_np]
        obstacle_points_tensor = np_to_tensor(points) if points is not None else None
        point_velocities_tensor = (
            np_to_tensor(velocities) if velocities is not None else None
        )

        # 调用PAN类，计算优化后的状态、速度和距离
        opt_state_tensor, opt_vel_tensor, opt_distance_tensor = self.pan(
            *nom_input_tensor, obstacle_points_tensor, point_velocities_tensor
        )

        # Tensor → NumPy
        opt_state_np, opt_vel_np = tensor_to_np(opt_state_tensor), tensor_to_np(
            opt_vel_tensor
        )

        self.cur_vel_array = opt_vel_np

        self.info["state_tensor"] = opt_state_tensor
        self.info["vel_tensor"] = opt_vel_tensor
        self.info["distance_tensor"] = opt_distance_tensor
        self.info['ref_state_tensor'] = nom_input_tensor[2]
        self.info['ref_speed_tensor'] = nom_input_tensor[3]

        self.info["ref_state_list"] = [
            state[:, np.newaxis] for state in nom_input_np[2].T
        ]
        self.info["opt_state_list"] = [state[:, np.newaxis] for state in opt_state_np.T]

        if self.check_stop():
            self.info["stop"] = True
            return np.zeros((2, 1)), self.info
        else:
            self.info["stop"] = False

        action = opt_vel_np[:, 0:1]

        return action, self.info

    def check_stop(self):
        return self.min_distance < self.collision_threshold
    

    def scan_to_point(
        self,
        state: np.ndarray,                             # 类型注解：state是NumPy数组
        scan: dict,
        scan_offset: list[float] = [0, 0, 0],          # 默认参数，传感器偏移量
        angle_range: list[float] = [-np.pi, np.pi],
        down_sample: int = 1,
    ) -> np.ndarray | None:                            # 返回值可能是NumPy数组或None
        
        """
        input:
            state: [x, y, theta]
            scan: {}                                              # 类型注解：scan是字典
                ranges: list[float], the range of the scan
                angle_min: float, the minimum angle of the scan
                angle_max: float, the maximum angle of the scan
                range_max: float, the maximum range of the scan
                range_min: float, the minimum range of the scan

            scan_offset: [x, y, theta], the relative position of the sensor to the robot state coordinate

        return point cloud: (2, n)                                # 返回值：点云，形状为(2, n)
        """
        point_cloud = []                                 # 存储有效的点云坐标

        ranges = np.array(scan["ranges"])                # 将扫描范围转换为NumPy数组
        angles = np.linspace(                            # 生成等间隔角度数组
            scan["angle_min"], 
            scan["angle_max"], 
            len(ranges)
        )

        for i in range(len(ranges)):
            scan_range = ranges[i]                        # 当前角度对应的距离值
            angle = angles[i]                             # 当前角度值

            # 检查距离是否在有效范围内
            if scan_range < (scan["range_max"] - 0.02) and scan_range > scan["range_min"]:
                # 检查角度是否在指定范围内
                if angle > angle_range[0] and angle < angle_range[1]:
                    # 极坐标 → 笛卡尔坐标
                    point = np.array(
                        [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                    )
                    point_cloud.append(point)

        # 无有效点时返回None
        if len(point_cloud) == 0:
            return None

        # 合并点云并转换坐标系
        point_array = np.hstack(point_cloud)                # 将列表转为(2, n)矩阵
        s_trans, s_R = get_transform(np.c_[scan_offset])    # 获取传感器坐标系的变换矩阵
        temp_points = s_R @ point_array + s_trans           # 将点云转换到传感器坐标系

        # 转换到全局坐标系
        trans, R = get_transform(state)                         # 获取机器人位姿的变换矩阵
        points = (R @ temp_points + trans)[:, ::down_sample]    # 降采样

        return points

    def scan_to_point_velocity(
        self,
        state,
        scan,
        scan_offset=[0, 0, 0],
        angle_range=[-np.pi, np.pi],
        down_sample=1,
    ):
        """
        input:
            state: [x, y, theta]
            scan: {}
                ranges: list[float], the ranges of the scan
                angle_min: float, the minimum angle of the scan
                angle_max: float, the maximum angle of the scan
                range_max: float, the maximum range of the scan
                range_min: float, the minimum range of the scan
                velocity: list[float], the velocity of the scan

            scan_offset: [x, y, theta], the relative position of the sensor to the robot state coordinate

        return point cloud: (2, n)
        """
        point_cloud = []
        velocity_points = []                # 存储有效点云的速度

        ranges = np.array(scan["ranges"])
        angles = np.linspace(
            scan["angle_min"], 
            scan["angle_max"], 
            len(ranges)
        )
        scan_velocity = scan.get("velocity", np.zeros((2, len(ranges))))

        # lidar_state = self.lidar_state_transform(state, np.c_[self.lidar_offset])
        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan["range_max"] - 0.02) and scan_range >= scan["range_min"]:
                if angle > angle_range[0] and angle < angle_range[1]:
                    point = np.array(
                        [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                    )
                    point_cloud.append(point)
                    velocity_points.append(scan_velocity[:, i : i + 1])         # 保存对应的障碍物速度

        if len(point_cloud) == 0:
            return None, None

        point_array = np.hstack(point_cloud)
        s_trans, s_R = get_transform(np.c_[scan_offset])
        temp_points = s_R.T @ (
            point_array - s_trans
        )  # points in the robot state coordinate

        trans, R = get_transform(state)
        points = (R @ temp_points + trans)[:, ::down_sample]

        velocity = np.hstack(velocity_points)[:, ::down_sample]

        return points, velocity


    # 触发DUNE训练
    def train_dune(self):
        self.pan.dune_layer.train_dune(self.dune_train_kwargs)


    def reset(self):
        self.ipath.point_index = 0
        self.ipath.curve_index = 0
        self.ipath.arrive_flag = False
        self.info["stop"] = False
        self.info["arrive"] = False
        self.cur_vel_array = np.zeros_like(self.cur_vel_array)

    def set_initial_path(self, path):

        '''
        set the initial path from the given path
        path: list of [x, y, theta, gear] 4x1 vector, gear is -1 (back gear) or 1 (forward gear)
        '''

        self.ipath.set_initial_path(path)

    # 根据给定的状态（位置和方向）初始化路径
    def set_initial_path_from_state(self, state):
        """
        Args:
            states: [x, y, theta] or 3x1 vector

        """
        self.ipath.init_check(state)
    
    # 设置机器人的参考速度（目标速度）
    def set_reference_speed(self, speed: float):

        """
        Args:
            speed: float, the reference speed of the robot
        """

        self.ipath.ref_speed = speed
        self.ref_speed = speed
    
    # 根据给定的起点和终点状态，更新初始路径
    def update_initial_path_from_goal(self, start, goal):

        """
        Args:
            start: [x, y, theta] or 3x1 vector
            goal: [x, y, theta] or 3x1 vector
        """

        self.ipath.update_initial_path_from_goal(start, goal)


    # 通过一组路径点（waypoints）更新机器人的初始路径
    def update_initial_path_from_waypoints(self, waypoints):

        """
        Args:
            waypoints: list of [x, y, theta] or 3x1 vector
        """

        self.ipath.set_ipath_with_waypoints(waypoints)


    # 动态调整路径规划或运动控制中的关键参数
    def update_adjust_parameters(self, **kwargs):

        """
        update the adjust parameters value: q_s, p_u, eta, d_max, d_min

        Args:
            q_s: float, the weight of the state cost                 值越大路径越贴近参考轨迹
            p_u: float, the weight of the speed cost                 值越大速度变化越平滑
            eta: float, the weight of the collision avoidance cost   值越大对障碍物越敏感
            d_max: float, the maximum distance to the obstacle      
            d_min: float, the minimum distance to the obstacle
        """
        
        self.pan.nrmp_layer.update_adjust_parameters_value(**kwargs)

    #  将方法转为属性调用形式（obj.min_distance 而非 obj.min_distance()）
    @property
    def min_distance(self):
        return self.pan.min_distance
    
    @property
    def dune_points(self):
        return self.pan.dune_points
    
    @property
    def nrmp_points(self):
        return self.pan.nrmp_points
    
    @property
    def initial_path(self):
        return self.ipath.initial_path
    
    @property
    def adjust_parameters(self):
        return self.pan.nrmp_layer.adjust_parameters
    
    @property
    def waypoints(self):

        '''
        Waypoints for generating the initial path
        '''

        return self.ipath.waypoints
    
    # 提供对MPC优化轨迹的只读访问，返回状态序列（预测时域内的状态）
    @property
    def opt_trajectory(self):

        '''
        MPC receding horizon trajectory under the velocity sequence
        return a list of state sequence, each state is a 3x1 vector
        '''

        return self.info["opt_state_list"]
    
    @property
    def ref_trajectory(self):

        '''
        Reference trajectory on the initial path
        return a list of state sequence, each state is a 3x1 vector
        '''

        return self.info["ref_state_list"]

    

    

