"""
PAN is the core class for the NeuPan algorithm. It is a proximal alternating-minimization network, consisting of NRMP and DUNE, that solves the optimization problem with numerous point-level collision avoidance constraints in each step. 
通过交替优化 NRMP 和 DUNE 模块解决带碰撞约束的优化问题

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han

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
"""

import torch
from neupan.blocks import NRMP, DUNE
import numpy as np
from math import inf
from typing import Optional
import numpy as np
from neupan.configuration import to_device, tensor_to_np
from neupan.util import downsample_decimation

class PAN(torch.nn.Module):         # 继承自 torch.nn.Module，是一个 PyTorch 神经网络模块
    """
    Args:
        receding: int, the number of steps in the receding horizon.
        step_time: float, the time step in the MPC framework.
        robot: robot, the robot instance including the robot information.
        iter_num: int, the number of iterations in the PAN algorithm.
        dune_max_num: int, the maximum number of points considered in the DUNE model.
        nrmp_max_num: int, the maximum number of points considered in the NRMP model.
        dune_checkpoint: str, the checkpoint path for the DUNE model.
        iter_threshold: float, the threshold for the iteration to judge the convergence.
        adjust_kwargs: dict, the keyword arguments for the adjust class.
        train_kwargs: dict, the keyword arguments for the train class.
    """

    def __init__(
        self,
        receding=10,            # 预测时域长度
        step_time=0.1,
        robot=None,
        iter_num=2,             # 交替优化迭代次数
        dune_max_num=100,       # DUNE 模型中考虑的最大点数
        nrmp_max_num=10,        # NRMP 模型中考虑的最大点数
        dune_checkpoint=None,
        iter_threshold=0.1,     # 用于判断迭代收敛的阈值
        adjust_kwargs=dict(),
        train_kwargs=dict(),
        **kwargs,
    ) -> None:
        super(PAN, self).__init__()

        self.robot = robot
        self.T = receding
        self.dt = step_time

        self.iter_num = iter_num
        self.iter_threshold = iter_threshold

        # NRMP 模块初始化
        self.nrmp_layer = NRMP(
            receding,
            step_time,
            robot,
            nrmp_max_num,
            eta=adjust_kwargs.get("eta", 10.0),             # 使用 adjust_kwargs.get("key", default) 语法安全获取参数
            d_max=adjust_kwargs.get("d_max", 1.0),
            d_min=adjust_kwargs.get("d_min", 0.1),
            q_s=adjust_kwargs.get("q_s", 1.0),
            p_u=adjust_kwargs.get("p_u", 1.0),
            ro_obs=adjust_kwargs.get("ro_obs", 400),
            bk=adjust_kwargs.get("bk", 0.1),
            solver=adjust_kwargs.get("solver", "ECOS"),
        )

        # DUNE 模块条件初始化
        self.no_obs = (nrmp_max_num == 0 or dune_max_num == 0)   # 无障碍模式标志
        self.nrmp_max_num = nrmp_max_num
        self.dune_max_num = dune_max_num

        if not self.no_obs:
            self.dune_layer = DUNE(   # 初始化DUNE模块
                receding,
                dune_checkpoint,
                robot,
                dune_max_num,
                train_kwargs,
            )
        else:
            self.dune_layer = None    # 无障碍时不初始化

        self.current_nom_values = [
            None,
            None,
            None,
            None,
        ]  # nom_s, nom_u, nom_lam, nom_mu

        self.printed = False

    def forward(
        self, 
        nom_s: torch.Tensor,                        # 名义状态 (3, receding+1)
        nom_u: torch.Tensor,                        # 名义控制 (2, receding)
        ref_s: torch.Tensor,                        # 参考轨迹（3，receding+1）
        ref_us: torch.Tensor,                       # 参考速度 (receding,)
        obs_points: torch.Tensor = None,            #（2，观测点数量），全局坐标
        point_velocities: torch.Tensor = None       # 障碍速度 (2, 观测点数量）)
    ):
        """
        input:
            - nom_s: nominal state; (3, receding+1) 
            - nom_u: nominal control; (2, receding)
            - ref_states: reference trajectory; (3, receding+1)
            - ref_us: reference speed array;  (receding,)
            - obs_points: (2, number of obs points), point cloud, global coordinate
            - velocities: (2, number of obs points), velocity of each obs point

        output:
            - opt_vel: optimal velocity tensor; (2, receding)
            - opt_state: optimal state array  (3, receding+1)

        process:
        """
        # 交替优化循环
        for i in range(self.iter_num):
            # 1. 障碍物处理
            if obs_points is not None and not self.no_obs:
                point_flow_list, R_list, obs_points_list = self.generate_point_flow(   # 生成障碍点流
                    nom_s, obs_points, point_velocities
                )
                mu_list, lam_list, sort_point_list = self.dune_layer(                  # DUNE前向传播 (核心碰撞约束处理)
                    point_flow_list, R_list, obs_points_list
                )
            else:
                mu_list, lam_list, sort_point_list = [], [], []
                
            # 2. NRMP轨迹优化
            nom_s, nom_u, nom_distance = self.nrmp_layer(
                nom_s, nom_u, ref_s, ref_us, mu_list, lam_list, sort_point_list
            )

            # 3. 收敛检查
            if self.stop_criteria(nom_s, nom_u, mu_list, lam_list):
                break

        return nom_s, nom_u, nom_distance


    def generate_point_flow(
        self, 
        nom_s: torch.Tensor, 
        obs_points: torch.Tensor, 
        point_velocities: Optional[torch.Tensor]=None
    ):

        '''
        generate the point flow (robot coordinate), rotation matrix and obs points (global coordinate) list in each receding step

        Args:
            nom_s: (3, receding+1)
            obs_points: (2, n)
            point_velocities: (2, n), x,y vel

        Returns:
            point_flow_list: list of (2, n); robot coordinate
            R_list: list of (2, 2); rotation matrix
            obs_points_list: list of (2, n); global coordinate
        '''

        # down sample the obs points by dune max num 

        if point_velocities is None:
            point_velocities = torch.zeros_like(obs_points)

        # 降采样处理
        if obs_points.shape[1] > self.dune_max_num:
            self.print_once(f"down sample the obs points from {obs_points.shape[1]} to {self.dune_max_num}") 
            obs_points = downsample_decimation(obs_points, self.dune_max_num)
            point_velocities = downsample_decimation(point_velocities, self.dune_max_num)

        
        obs_points_list = []   # 存储全局坐标系下的障碍物点位置,包含从当前时刻到预测时域结束的所有时刻的障碍物位置
        point_flow_list = []   # 存储机器人坐标系下的障碍物点位置,包含从当前时刻到预测时域结束的所有时刻的障碍物位置
        R_list = []            # 存储每个时刻的旋转矩阵

        if point_velocities is None:
            point_velocities = torch.zeros_like(obs_points)

        # 时间步循环处理
        for i in range(self.T+1):
            # 预测未来时域内障碍物的位置
            receding_obs_points = obs_points + i * (point_velocities * self.dt)
            obs_points_list.append(receding_obs_points) 
            
            # 坐标系转换
            p0, R = self.point_state_transform(nom_s[:, i], receding_obs_points)
            point_flow_list.append(p0)
            R_list.append(R)

        return point_flow_list, R_list, obs_points_list


    def point_state_transform(self, state: torch.Tensor, obs_points: torch.Tensor):

        '''
        transform the position of obstacle points to the robot coordinate system in each receding step
        
        input: 
            state: [x, y, theta] -- transition and rotation matrix
            obs_points: (2, n) -- point cloud

        output:
            p0: (2, n) point cloud in the robot coordinate system
            R: (2, 2) rotation matrix
        '''

        state = state.reshape((3, 1))
        trans = state[0:2]    # 机器人在全局坐标系中的位置
        theta = state[2, 0]   # 机器人的朝向角（相对于全局坐标系的X轴）
        R = to_device(torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]))

        p0 = R.T @ (obs_points - trans)   # obs_points - trans: 将障碍物点减去机器人的位置,当于将坐标系原点移动到机器人位置

        return p0, R
    

    def stop_criteria(self, nom_s, nom_u, mu_list, lam_list):

        '''
        stop criteria for the iteratio
       
        DUNE层:处理障碍物约束,输出权重mu_list和lam_list
        NRMP层:基于约束进行轨迹优化,输出状态nom_s和控制nom_u
        '''

        # 首次迭代存储当前值
        if self.current_nom_values[0] is None:
            self.current_nom_values = [nom_s, nom_u, mu_list, lam_list]
            return False
        
        else:
            # 计算状态变化量
            nom_s_diff = torch.norm(nom_s - self.current_nom_values[0])
            nom_u_diff = torch.norm(nom_u - self.current_nom_values[1])

            if len(mu_list) == 0 or len(self.current_nom_values[2]) == 0:
                diff = nom_s_diff**2 + nom_u_diff**2

            else:
                effect_num = min([mu_list[0].shape[1], self.current_nom_values[2][0].shape[1], self.nrmp_max_num])

                # 计算约束变化量
                mu_diff = torch.norm( (torch.cat(mu_list)[:, :effect_num] - torch.cat(self.current_nom_values[2])[:, :effect_num] )) / effect_num
                lam_diff = torch.norm( (torch.cat(lam_list)[:, :effect_num]  - torch.cat(self.current_nom_values[3])[:, :effect_num]  )) / effect_num
                
                # 综合收敛指标
                diff = mu_diff**2 + lam_diff**2

            # 更新并返回判断结果
            self.current_nom_values = [nom_s, nom_u, mu_list, lam_list]
            return diff < self.iter_threshold

    @property
    # 获取机器人到障碍物的最小距离
    def min_distance(self):

        if self.dune_layer is None or self.no_obs:
            return inf
        
        else:
            return self.dune_layer.min_distance
        
    @property
    # 获取DUNE层处理后的障碍物点
    def dune_points(self):
        
        if self.dune_layer is None or self.no_obs:
            return None
        else:
            return tensor_to_np(self.dune_layer.points)


    @property
    # 获取NRMP层处理后的障碍物点
    def nrmp_points(self):
        if self.nrmp_layer is None or self.no_obs:
            return None
        else:
            return tensor_to_np(self.nrmp_layer.points)

    @property
    def min_distance(self):
        return inf if self.no_obs else self.dune_layer.min_distance
    
    # 确保某条消息只打印一次
    def print_once(self, message):
        if not self.printed:
            print(message)
            self.printed = True
