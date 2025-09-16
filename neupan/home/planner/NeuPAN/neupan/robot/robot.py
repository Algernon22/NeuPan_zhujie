'''
robot class define the robot model and the kinematics model for NeuPAN. It also generate the constraints and cost functions for the optimization problem.

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

from math import inf
import numpy as np
from typing import Optional, Union  # Optional表示参数可以是给定的类型或None; Union 表示参数可以是多种类型中的一种
import cvxpy as cp
from math import sin, cos, tan
import torch
from neupan.configuration import to_device 
from neupan.util import gen_inequal_from_vertex

class robot:

    def __init__(
        self,
        receding: int = 10,                                         # 时域长度
        step_time: float = 0.1,                                     # 每步的时间间隔
        kinematics: Optional[str] = None,                           # 运动学模型类型
        vertices: Optional[Union[list[float], np.ndarray]] = None,  # 机器人形状顶点
        max_speed: list[float] = [inf, inf],                        # 最大速度 [线速度, 角速度]
        max_acce: list[float] = [inf, inf],                         # 最大加速度 [线加速度, 角加速度]
        wheelbase: Optional[float] = None,                          # 阿克曼模型的轴距
        length: Optional[float] = None,                             # 机器人长度
        width: Optional[float] = None,                              # 机器人宽度
        **kwargs,
    ):
        
        if kinematics is None:
            raise ValueError("kinematics is required")

        self.shape = None

        # 顶点计算（机器人几何形状）
        self.vertices = self.cal_vertices(vertices, length, width, wheelbase)
        
        # 生成半平面表示的不等式约束（机器人凸包）
        self.G, self.h = gen_inequal_from_vertex(self.vertices)

        self.T = receding
        self.dt = step_time
        self.L = wheelbase
        self.kinematics = kinematics

        # np.c_将列表转换为列向量; 检查输入是否为list  T:列表 → 二维列向量  F:保持原格式
        self.max_speed = np.c_[max_speed] if isinstance(max_speed, list) else max_speed
        self.max_acce = np.c_[max_acce] if isinstance(max_acce, list) else max_acce

        if kinematics == 'acker':
            if self.max_speed[1] >= 1.57:
                print(f"Warning: max steering angle of acker robot is {self.max_speed[1]} rad, which is larger than 1.57 rad, so it is limited to 1.57 rad")
                self.max_speed[1] = 1.57

        self.speed_bound = self.max_speed          # 速度约束处理
        self.acce_bound = self.max_acce * self.dt  # 加速度约束转换为速度变化约束

        self.name = kwargs.get("name", self.kinematics + "_robot" + '_default') 


    def define_variable(self, no_obs: bool = False, indep_dis: cp.Variable = None):

        """
        define variables
        定义路径跟随状态，输入控制，安全距离三个优化变量
        """

        self.indep_s = cp.Variable((3, self.T + 1), name="state")  # 状态变量 (x, y, θ) t0 - T
        self.indep_u = cp.Variable((2, self.T), name="vel")  # # 控制变量 (v, ω) t1 - T
        
        indep_list = (
            [self.indep_s, self.indep_u]
            if no_obs
            else [self.indep_s, self.indep_u, indep_dis]
        )

        return indep_list

    def state_parameter_define(self):

        '''
        state parameters:
            - para_gamma_a: q*reference state, 3 * (T+1)
            - para_gamma_b: p*reference speed array, T
            - para_s: nominal state, 3 * (T+1)
            - para_A_list, para_B_list, para_C_list: for state transition model

            此方法定义优化问题中的参数(Parameters)，这些参数在优化问题构建后可以动态更新，而无需重新构建整个问题。
        '''

        # 状态参考参数
        self.para_s = cp.Parameter((3, self.T+1), name='para_state')          # 表示规划时域内的参考状态轨迹
        self.para_gamma_a = cp.Parameter((3, self.T+1), name='para_gamma_a')  # 状态跟踪代价的加权参考点
        self.para_gamma_b = cp.Parameter((self.T,), name='para_gamma_b')      # 速度跟踪代价的加权参考点

        # 动力学线性化参数
        self.para_A_list = [ cp.Parameter((3, 3), name='para_A_'+str(t)) for t in range(self.T)]  # T个状态转移雅可比矩阵
        self.para_B_list = [ cp.Parameter((3, 2), name='para_B_'+str(t)) for t in range(self.T)]  # T个控制输入雅可比矩阵
        self.para_C_list = [ cp.Parameter((3, 1), name='para_C_'+str(t)) for t in range(self.T)]  # T个线性化过程中的常数项

        return [self.para_s, self.para_gamma_a, self.para_gamma_b] + self.para_A_list + self.para_B_list + self.para_C_list  # 列表拼接 [ , , ]+list1+list2+list3


    def coefficient_parameter_define(self, no_obs: bool = False, max_num: int = 10):

        """
        gamma_c: lam.T
        zeta_a: lam.T @ p + mu.T @ h
        """

        if no_obs:
            self.para_gamma_c, self.para_zeta_a = [], []

        else:
            self.para_gamma_c = [  # 障碍物约束的法向量矩阵
                cp.Parameter(
                    (max_num, 2),
                    value=np.zeros((max_num, 2)),
                    name="para_gamma_c" + str(i),
                )
                for i in range(self.T)
            ]  # lam.T, fa
            self.para_zeta_a = [  # 障碍物约束的边界值
                cp.Parameter(
                    (max_num, 1),
                    value=np.zeros((max_num, 1)),
                    name="para_zeta_a" + str(i),
                )
                for i in range(self.T)
            ]  # lam.T @ p + mu.T @ h, fb

        return self.para_gamma_c + self.para_zeta_a


    def C0_cost(self, para_p_u, para_q_s):
        
        '''
        reference state cost and control vector cost
        状态代价：当前状态与参考状态的偏差
        控制代价：控制输入与参考速度的偏差

        para_p_u: weight of speed cost
        para_q_s: weight of state cost
        '''

        diff_u = para_p_u * self.indep_u[0, :] - self.para_gamma_b  # 控制代价：当前速度与参考速度的偏差
        diff_s = para_q_s * self.indep_s - self.para_gamma_a        # 状态代价：当前状态与参考状态/预设的全局参考轨迹的偏差

        C0_cost = cp.sum_squares(diff_s) + cp.sum_squares(diff_u)   # 状态代价和控制代价的平方和

        return C0_cost

    def proximal_cost(self):

        """
        proximal cost
        """

        proximal_cost = cp.sum_squares(self.indep_s - self.para_s)  # 当前状态与参考状态/局部轨迹的偏差平方和

        return proximal_cost


    def I_cost(self, indep_dis, ro_obs):

        cost = 0
        indep_t = self.indep_s[0:2, 1:]  # 跳过初始时刻,提取状态变量中的位置分量（x, y）

        I_list = []

        for t in range(self.T):
            # 减号前: 结果是一个 (max_num, 1) 的向量, 每个元素表示机器人位置在特定障碍物点法向量上的投影;
            # 减号后: 障碍物安全边界的松弛项
            # 机器人位置与障碍物边界的距离减去安全裕度
            I_dpp = self.para_gamma_c[t] @ indep_t[:, t:t+1] - self.para_zeta_a[t] - indep_dis[0, t]  
            I_list.append(I_dpp)

        I_array = cp.vstack(I_list)  # 垂直堆叠
        cost += 0.5 * ro_obs * cp.sum_squares(cp.neg(I_array))  # neg(x) = max(-x, 0), 当 I_dpp < 0（约束违反）：neg(I_dpp) = -I_dpp > 0 实现仅惩罚约束违反部分

        return cost

    def dynamics_constraint(self):

        '''
        linear dynamics constraints: x_{t+1} = A_t @ x_t + B_t @ u_t + C_t
        '''

        temp_list = []

        for t in range(self.T):
            indep_st = self.indep_s[:, t:t+1]  # 提取当前时刻状态变量
            indep_ut = self.indep_u[:, t:t+1]  # 提取当前时刻控制变量

            # dynamic constraints
            A = self.para_A_list[t]  # 提取状态转移矩阵A
            B = self.para_B_list[t]  # 提取控制输入矩阵B
            C = self.para_C_list[t]  # 提取常数项矩阵C
            
            # 计算下一时刻的预测状态
            temp_list.append(A @ indep_st + B @ indep_ut + C)
        
        # 整个约束表示为: [x₁, x₂, ..., x_T] = [A₀x₀+B₀u₀+C₀, A₁x₁+B₁u₁+C₁, ..., A_{T-1}x_{T-1}+B_{T-1}u_{T-1}+C_{T-1}]
        constraints = [ self.indep_s[:, 1:] == cp.hstack(temp_list) ]

        return constraints 


    def bound_su_constraints(self):

        '''
        bound constraints on init state, speed, and acceleration   
        '''

        constraints = []

        constraints += [ cp.abs(self.indep_u[:, 1:] - self.indep_u[:, :-1] ) <= self.acce_bound ] 
        constraints += [ cp.abs(self.indep_u) <= self.speed_bound]
        constraints += [ self.indep_s[:, 0:1] == self.para_s[:, 0:1] ]  # 优化起始于当前实际状态

        return constraints
    

    def generate_state_parameter_value(self, nom_s, nom_u, qs_ref_s, pu_ref_us):

        state_value_list = [nom_s, qs_ref_s, pu_ref_us]

        tensor_A_list = []
        tensor_B_list = []
        tensor_C_list = []

        for t in range(self.T):
            nom_st = nom_s[:, t:t+1]
            nom_ut = nom_u[:, t:t+1]

            if self.kinematics == 'acker':
                A, B, C = self.linear_ackermann_model(nom_st, nom_ut, self.dt, self.L)
            elif self.kinematics == 'diff':
                A, B, C = self.linear_diff_model(nom_st, nom_ut, self.dt)
            else:
                raise ValueError('kinematics currently only supports acker or diff')

            tensor_A_list.append(A)
            tensor_B_list.append(B)
            tensor_C_list.append(C)

        state_value_list += tensor_A_list
        state_value_list += tensor_B_list
        state_value_list += tensor_C_list

        return state_value_list


    
    def linear_ackermann_model(self, nom_st, nom_ut, dt, L):
        
        phi = nom_st[2, 0]
        v, psi = nom_ut[0, 0], nom_ut[1, 0]

        A = torch.Tensor([ [1, 0, -v * dt * sin(phi)], [0, 1, v * dt * cos(phi)], [0, 0, 1] ])

        B = torch.Tensor([ [cos(phi)*dt, 0], [sin(phi)*dt, 0], 
                        [ tan(psi)*dt / L, v*dt/(L * (cos(psi))**2 ) ] ])

        C = torch.Tensor([ [ phi*v*sin(phi)*dt ], [ -phi*v*cos(phi)*dt ], 
                        [ -psi * v*dt / ( L * (cos(psi))**2) ]])
        

        return to_device(A), to_device(B), to_device(C)   
    

    def linear_diff_model(self, nom_state, nom_u, dt):
        
        phi = nom_state[2, 0]
        v = nom_u[0, 0]

        A = torch.Tensor([ [1, 0, -v * dt * sin(phi)], [0, 1, v * dt * cos(phi)], [0, 0, 1] ])

        B = torch.Tensor([ [cos(phi)*dt, 0], [sin(phi)*dt, 0], 
                        [ 0, dt ] ])

        C = torch.Tensor([ [ phi*v*sin(phi)*dt ], [ -phi*v*cos(phi)*dt ], 
                        [ 0 ]])
                
        return to_device(A), to_device(B), to_device(C) 



    def cal_vertices_from_length_width(self, length, width, wheelbase=None):
        """
        Calculate initial vertices of a rectangle representing a robot.

        Args:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.
            wheelbase (float): Wheelbase of the robot.

        Returns:
            vertices (np.ndarray): Vertices of the rectangle. shape: (2, 4)
        """
        wheelbase = 0 if wheelbase is None else wheelbase

        start_x = -(length - wheelbase) / 2
        start_y = -width / 2

        point0 = np.array([[start_x], [start_y]])  # left bottom point
        point1 = np.array([[start_x + length], [start_y]])
        point2 = np.array([[start_x + length], [start_y + width]])
        point3 = np.array([[start_x], [start_y + width]])

        return np.hstack((point0, point1, point2, point3))
    
    def cal_vertices(self, vertices = None, length = None, width = None, wheelbase=None):

        '''
        Generate vertices. If vertices is not set, generate vertices from length, width, and wheelbase.

        Args:
            vertices: list of vertices or numpy array of vertices, [[x1, y1], [x2, y2], ...] or (2, N)
            length: length of the robot
            width: width of the robot
            wheelbase: wheelbase of the robot

        Returns:
            vertices_np: numpy array of vertices, shape: (2, N), N >3
        '''

        if vertices is not None:
           if isinstance(vertices, list):
                vertices_np = np.array(vertices).T

           elif isinstance(vertices, np.ndarray):
                vertices_np = vertices
           else:
                raise ValueError("vertices must be a list or numpy array")
           
        else:
            self.shape = "rectangle"
            vertices_np = self.cal_vertices_from_length_width(length, width, wheelbase)
            self.length = length
            self.width = width
            self.wheelbase = wheelbase

        assert vertices_np.shape[1] >= 3, "vertices must be a numpy array of shape (2, N), N >= 3"

        return vertices_np

