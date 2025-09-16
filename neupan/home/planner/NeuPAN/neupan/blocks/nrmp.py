"""
NRMP (Neural Regularized Motion Planner) is the core class of the PAN class. It solves the optimization problem integrating the neural latent distance space to generate the optimal control sequence.

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
"""

import torch
import cvxpy as cp
import numpy as np
from neupan.robot import robot
from neupan.configuration import to_device, value_to_tensor, np_to_tensor
from cvxpylayers.torch import CvxpyLayer
from neupan.util import time_it
from typing import Optional, List


class NRMP(torch.nn.Module):

    def __init__(
        self,
        receding: int,
        step_time: float,
        robot: robot,
        nrmp_max_num: int = 10, # NRMP 模型中考虑的最大点云数
        eta: float = 10.0,      # 安全距离权重参数
        d_max: float = 1.0,     # 最大安全距离
        d_min: float = 0.1,     # 最小安全距离
        q_s: float = 1.0,       # 状态权重参数
        p_u: float = 1.0,       # 控制权重参数
        ro_obs: float = 400,    # 障碍物惩罚权重
        bk: float = 0.1,        # 收敛系数
        **kwargs,
    ) -> None:
        super(NRMP, self).__init__()

        self.T = receding
        self.dt = step_time
        self.robot = robot
        self.G = np_to_tensor(robot.G)
        self.h = np_to_tensor(robot.h)

        self.max_num = nrmp_max_num
        self.no_obs = False if nrmp_max_num > 0 else True

        # adjust parameters
        self.eta = value_to_tensor(eta, True)
        self.d_max = value_to_tensor(d_max, True)
        self.d_min = value_to_tensor(d_min, True)
        self.q_s = value_to_tensor(q_s, True)
        self.p_u = value_to_tensor(p_u, True)

        self.ro_obs = ro_obs
        self.bk = bk

        # 根据有无障碍物来创建不同的可变参数列表
        self.adjust_parameters = (
            [self.q_s, self.p_u]
            if self.no_obs
            else [self.q_s, self.p_u, self.eta, self.d_max, self.d_min]
        )

        self.variable_definition()   # 定义优化变量
        self.parameter_definition()  # 定义优化参数
        self.problem_definition()    # 定义优化参数

        self.obstacle_points = None
        self.solver = kwargs.get("solver", "ECOS") 

    @time_it("- nrmp forward")
    def forward(
        self,
        nom_s: torch.Tensor,                                 # 标称状态 (3 × (T+1))
        nom_u: torch.Tensor,                                 # 标称速度 (1 × T)
        ref_s: torch.Tensor,                                 # 参考状态 (3 × (T+1))
        ref_us: torch.Tensor,                                # 参考速度 (T,)
        mu_list: Optional[List[torch.Tensor]] = None,        # [T+1, (num_obs, edge_dim)],每个障碍物点对各个约束边的权重
        lam_list: Optional[List[torch.Tensor]] = None,       # [T+1, (num_obs, 2)],障碍物点在状态空间中的梯度方向
        point_list: Optional[List[torch.Tensor]] = None,     # [T+1, (num_obs, 2)]障碍物点在全局坐标系下的位置
    ):
        """
        nom_s: nominal state, 3 * (T+1)
        nom_u: nominal speed, 1 * T
        ref_s: reference state, 3 * (T+1)
        ref_us: reference speed array, (T,),
        mu_list: list of mu matrix, (max_num, )
        lam_list: list of lam matrix, (max_num, 1)
        point_list: list of obstacle points, (max_num, 2)
        """

        # 障碍点处理
        if point_list:
            self.obstacle_points = point_list[0][
                :, : self.max_num
            ]  # current obstacle points considered in the optimization,取列表中的第一个张量,取所有行的前 max_num 列

        
        parameter_values = self.generate_parameter_value(
            nom_s, nom_u, ref_s, ref_us, mu_list, lam_list, point_list
        )

        '''
        *parameter_values: 将 parameter_values 按顺序绑定到 self.para_list 中的参数 -- 给优化问题的"空壳"填充具体数值
        访问 CvxpyLayer 的 call 方法, 使用初始化时构建的优化问题
        返回解的顺序严格按照 variables=self.indep_list 中定义的顺序, 依次是状态，速度，距离
        '''
        solutions = self.nrmp_layer(*parameter_values, solver_args={"solve_method": self.solver}) 
        opt_solution_state = solutions[0]    
        opt_solution_vel = solutions[1]      
        nom_d = None if self.no_obs else solutions[2]  

        return opt_solution_state, opt_solution_vel, nom_d

    def generate_parameter_value(
        self, nom_s, nom_u, ref_s, ref_us, mu_list, lam_list, point_list
    ):
        
        adjust_value_list = self.generate_adjust_parameter_value()

        # 获取机器人状态参数值
        state_value_list = self.robot.generate_state_parameter_value(
            nom_s, nom_u, self.q_s * ref_s, self.p_u * ref_us
        )

        # 获取障碍物系数参数值
        coefficient_value_list = self.generate_coefficient_parameter_value(
            mu_list, lam_list, point_list
        )

        return state_value_list + coefficient_value_list + adjust_value_list

    def generate_adjust_parameter_value(self):
        return self.adjust_parameters

    def update_adjust_parameters_value(self, **kwargs):
        '''
        update the adjust parameters value: q_s, p_u, eta, d_max, d_min
        '''

        # 尝试从kwargs获取指定键值, 若键不存在，则使用当前属性值
        self.q_s = value_to_tensor(kwargs.get("q_s", self.q_s), True)
        self.p_u = value_to_tensor(kwargs.get("p_u", self.p_u), True)
        self.eta = value_to_tensor(kwargs.get("eta", self.eta), True)
        self.d_max = value_to_tensor(kwargs.get("d_max", self.d_max), True)
        self.d_min = value_to_tensor(kwargs.get("d_min", self.d_min), True)

        self.adjust_parameters = (
            [self.q_s, self.p_u]
            if self.no_obs
            else [self.q_s, self.p_u, self.eta, self.d_max, self.d_min]
        )


    def generate_coefficient_parameter_value(self, mu_list, lam_list, point_list):
        """
        generate the parameters values for obstacle point avoidance

        Args:
            mu_list: list of mu matrix,
            lam_list: list of lam matrix,
            point_list: list of sorted obstacle points,

        Returns:
            fa_list: list of fa matrix,
            fb_list: list of fb matrix,
        """

        # 无障碍物处理
        if self.no_obs:
            return []
        else:
            fa_list = [to_device(torch.zeros((self.max_num, 2))) for t in range(self.T)]   # 时间步列表，长度为T,每个元素是PyTorch张量，形状为 (self.max_num, 2) - 每个障碍物点有两个系数
            fb_list = [to_device(torch.zeros((self.max_num, 1))) for t in range(self.T)]   # 时间步列表，长度为T,每个元素是PyTorch张量，形状为 (self.max_num, 1) - 每个障碍物点有一个系数

            # DUNE层传入空mu_list
            if not mu_list:
                return fa_list + fb_list
            else:
                for t in range(self.T):
                    # 获取当前时刻的mu, lam, point
                    mu, lam, point = mu_list[t + 1], lam_list[t + 1], point_list[t + 1]   
                    fa = lam.T  # fa: 约束法向量,指示机器人应远离障碍物的方向
                    
                    # temp: 当前障碍物点在约束方向上的投影
                    temp = (
                        torch.bmm(lam.T.unsqueeze(1), point.T.unsqueeze(2))
                    ).squeeze(1)
                    # fb: 安全距离的边界值
                    fb = temp + mu.T @ self.h   # μ^T * h: 障碍物安全边界的松弛项 

                    '''
                    fa,fb 的物理意义:
                        (机器人位置-障碍物位置)fa <= fb
                        即机器人与障碍物的距离大于安全阈值

                    lam.T.unsqueeze(1)：将形状 (2, num_obs) → (2, 1, num_obs)
                    point.T.unsqueeze(2)：将形状 (2, num_obs) → (2, num_obs, 1)
                    bmm: 批矩阵乘法: (2,1,num_obs) @ (2,num_obs,1) → (2,1,1)
                    squeeze(1)  # 将 (2,1,1) → (2,1)
                    ''' 

                    pn = min(mu.shape[1], self.max_num)  # 确定实际障碍物数量
                    
                    # 赋值实际障碍物参数
                    fa_list[t][:pn, :] = fa[:pn, :] # 取列表中第t个元素 → 二维张量 (max_num, 2),行范围：0 到 pn-1 列范围：: 所有列
                    fb_list[t][:pn, :] = fb[:pn, :]

                    # 无效位置填充: 用第一个障碍物填充剩余位置
                    fa_list[t][pn:, :] = fa[0, :] 
                    fb_list[t][pn:, :] = fb[0, :]

            return fa_list + fb_list   # 第一维结构一致，所以可拼接

    def variable_definition(self):
        self.indep_dis = cp.Variable(                  # cp.Variable：使用 CVXPY 库创建优化变量
            (1, self.T), name="distance", nonneg=True  # 变量形状为 1 行 × T 列（T是时间步数）nonneg=True: 所有元素非负
        )  # 从时间 t1 到 T 的距离序列
        
        self.indep_list = self.robot.define_variable(self.no_obs, self.indep_dis)

    def parameter_definition(self):

        self.para_list = []

        self.para_list += self.robot.state_parameter_define()
        self.para_list += self.robot.coefficient_parameter_define(
            self.no_obs, self.max_num
        )
        self.para_list += self.adjust_parameter_define()

    def problem_definition(self):
        """
        define the optimization problem
        只构建一次优化问题框架（变量和约束结构），但参数值留空
        """

        prob = self.construct_prob()

        self.nrmp_layer = to_device(
            CvxpyLayer(prob, parameters=self.para_list, variables=self.indep_list)
        )

    def construct_prob(self):

        nav_cost, nav_constraints = self.nav_cost_cons()
        dune_cost, dune_constraints = self.dune_cost_cons()

        if self.no_obs:
            prob = cp.Problem(cp.Minimize(nav_cost), nav_constraints)
        else:
            prob = cp.Problem(
                cp.Minimize(nav_cost + dune_cost), nav_constraints + dune_constraints
            )

        assert prob.is_dcp(dpp=True)

        return prob

    def adjust_parameter_define(self, **kwargs):
        """
        q and p: the weight of state and control loss, respectively
        eta, d_max, d_min: the parameters for safety distance
        """

        self.para_q_s = cp.Parameter(name="para_q_s", value=kwargs.get("q_s", 1.0))
        self.para_p_u = cp.Parameter(name="para_p_u", value=kwargs.get("p_u", 1.0))

        self.para_eta = cp.Parameter(
            value=kwargs.get("eta", 8), nonneg=True, name="para_eta"
        )
        self.para_d_max = cp.Parameter(
            name="para_d_max", value=kwargs.get("d_max", 1.0), nonneg=True
        )
        self.para_d_min = cp.Parameter(
            name="para_d_min", value=kwargs.get("d_min", 0.1), nonneg=True
        )

        adjust_para_list = (
            [self.para_q_s, self.para_p_u]
            if self.no_obs
            else [
                self.para_q_s,
                self.para_p_u,
                self.para_eta,
                self.para_d_max,
                self.para_d_min,
            ]
        )

        return adjust_para_list

    def nav_cost_cons(self):

        cost = 0
        constraints = []

        cost += self.robot.C0_cost(self.para_p_u, self.para_q_s)
        cost += 0.5 * self.bk * self.robot.proximal_cost()

        constraints += self.robot.dynamics_constraint()
        constraints += self.robot.bound_su_constraints()

        return cost, constraints

    def dune_cost_cons(self):

        cost = 0
        constraints = []

        cost += self.C1_cost_d()  # distance cost

        if not self.no_obs:
            cost += self.robot.I_cost(self.indep_dis, self.ro_obs)

        constraints += self.bound_dis_constraints()

        return cost, constraints

    def bound_dis_constraints(self):

        constraints = []

        constraints += [self.indep_dis >= self.para_d_min]
        constraints += [self.indep_dis <= self.para_d_max]
        # constraints += [cp.max(self.indep_dis) <= self.para_d_max]
        # constraints += [cp.min(self.indep_dis) >= self.para_d_min]

        return constraints

    def C1_cost_d(self):
        return -self.para_eta * cp.sum(self.indep_dis)

    @property
    def points(self):
        """
        point considered in the nrmp layer

        """

        return self.obstacle_points
