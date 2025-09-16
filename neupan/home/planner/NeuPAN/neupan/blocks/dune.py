"""
DUNE (Deep Unfolded Neural Encoder) is the core class of the PAN class. It maps the point flow to the latent distance space: mu and lambda. 
将点流(point flow)映射到潜在距离空间(latent distance space)，生成距离特征 μ 和 λ

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
from math import inf
from neupan.blocks import ObsPointNet, DUNETrain
from neupan.configuration import np_to_tensor, to_device
from neupan.util import time_it, file_check, repeat_mk_dirs
from typing import Optional
import sys

class DUNE(torch.nn.Module):

    def __init__(
        self, 
        receding: int=10, 
        checkpoint =None, 
        robot=None, 
        dune_max_num: int=100, 
        train_kwargs: dict=dict()
    ) -> None:   # -> None: 类型提示，表示方法返回None
        super(DUNE, self).__init__()
  
        self.T = receding
        self.max_num = dune_max_num

        # 约束矩阵G和向量h定义了机器人的安全区域 G*x ≤ h 表示点x在安全区域内
        self.robot = robot
        self.G = np_to_tensor(robot.G)      # 机器人的几何约束矩阵
        self.h = np_to_tensor(robot.h)      # 机器人的几何约束向量
        self.edge_dim = self.G.shape[0]     # 约束边数
        self.state_dim = self.G.shape[1]    # 状态维度

        self.model = to_device(ObsPointNet(2, self.edge_dim))
        self.load_model(checkpoint, train_kwargs)

        self.obstacle_points = None
        self.min_distance = inf


        
    @time_it('- dune forward')   # 打印函数执行时间
    def forward(
        self, 
        point_flow: list[torch.Tensor],             # T+1个时刻, 每时刻100个2D点, 三维矩阵
        R_list: list[torch.Tensor], 
        obs_points_list: list[torch.Tensor]=[]      # 障碍物点在全局坐标系下的表示,每个张量形状: (2, num_points)
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:  

        '''
        map point flow to the latent distance features: lam, mu

        Args:
            point_flow: point flow under the robot coordinate, list of (state_dim, num_points); list length: T+1
            R_list: list of Rotation matrix, list of (2, 2), used to generate the lam from mu; list length: T
            obstacle_points: tensor of shape (2, num_points), global coordinate; 

        Returns: 
            lam_list: list of lam tensor, each element is a tensor of shape (state_dim, num_points); list length: T+1
                λ是μ在状态空间中的表示
            mu_list: list of mu tensor, each element is a tensor of shape (edge_number, num_points); list length: T+1
                μ不是距离本身,表示一个障碍物点对各个约束边的"影响权重"
            sort_point_list: list of point tensor, each element is a tensor of shape (state_dim, num_points); list length: T+1; 
        
        Simulation
            # 输入
            point_flow = [
                torch.randn(2, 50),  # 时刻0
                torch.randn(2, 50),  # 时刻1
                torch.randn(2, 50)   # 时刻2
            ]

            # 处理流程
            1. 合并: total_points = (2, 150)
            2. 神经网络: total_mu = (4, 150)
            3. 循环3次: 
            - 时刻0: mu=(4,50), lam=(2,50), 排序
            - 时刻1: mu=(4,50), lam=(2,50), 排序
            - 时刻2: mu=(4,50), lam=(2,50), 排序

            # 输出
            mu_list = [(4,50), (4,50), (4,50)]    # 排序后
            lam_list = [(2,50), (2,50), (2,50)]   # 排序后
            sort_point_list = [(2,50), (2,50), (2,50)]  # 排序后
        '''

        mu_list, lam_list, sort_point_list = [], [], []
        self.obstacle_points = obs_points_list[0]        # 保存时刻0的障碍物点
        total_points = torch.hstack(point_flow)          # 水平堆叠所有时刻的点
        
        # map the point flow to the latent distance features mu
        with torch.no_grad():                          # 禁用梯度计算, 节省计算资源
            total_mu = self.model(total_points.T).T    # 把所有的障碍物点喂给模型, 输出每个点的距离特征mu
        
        for index in range(self.T+1):
            num_points = point_flow[index].shape[1]                     # 当前时刻的障碍物点数量
            mu = total_mu[:, index*num_points : (index+1)*num_points]   # 切片提取当前时刻的μ特征,时刻0取[:, 0:100]，时刻1取[:, 100:200]
            R = R_list[index]                                           # 当前时刻的旋转矩阵
            p0 = point_flow[index]                                      # 当前时刻的障碍物点在机器人坐标系下的表示
            lam = (- R @ self.G.T @ mu)                                 # 计算距离特征对应的λ权重 λ = -R × G^T × μ

            # 边界情况处理: 从(edge_dim,)变为(edge_dim, 1)
            if mu.ndim == 1:
                mu = mu.unsqueeze(1)
                lam = lam.unsqueeze(1)

            # 计算点到约束边界的"加权"距离
            distance = self.cal_objective_distance(mu, p0)

            if index == 0: 
                self.min_distance = torch.min(distance) 
            
            sort_indices = torch.argsort(distance)

            mu_list.append(mu[:, sort_indices])
            lam_list.append(lam[:, sort_indices])
            sort_point_list.append(obs_points_list[index][:, sort_indices])

        return mu_list, lam_list, sort_point_list

    # 计算障碍物点到机器人约束边界的加权距离
    def cal_objective_distance(self, mu: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:

        '''
        input: 
            mu: (edge_dim, num_points)
            p0: (state_dim, num_points)   障碍物点在机器人坐标系下的坐标  
        output:
            distance:  mu.T (G @ p0 - h),  (num_points,)
        ''' 

        temp = (self.G @ p0 - self.h).T.unsqueeze(2)        # 计算约束违反程度,结果为负，表示点在安全区域内；为正表示违反约束
        muT = mu.T.unsqueeze(1)
        
        distance = torch.squeeze(torch.bmm(muT, temp)) 

        if distance.ndim == 0:
            distance = distance.unsqueeze(0)

        return distance
    


    def load_model(self, checkpoint: Optional[str]=None, train_kwargs: Optional[dict]=None):

        '''
        checkpoint: pth file path of the model
        '''

        try:
            if checkpoint is None:
                raise FileNotFoundError

            self.abs_checkpoint_path = file_check(checkpoint)
            self.model.load_state_dict(torch.load(self.abs_checkpoint_path, map_location=torch.device('cpu')))
            to_device(self.model)
            self.model.eval()

        except FileNotFoundError:

            if train_kwargs is None or len(train_kwargs) == 0:
                print('No train kwargs provided. Default value will be used.')
                train_kwargs = dict()
            
            direct_train = train_kwargs.get('direct_train', False)

            if direct_train:
                print('train or test the model directly.')
                return 

            if self.ask_to_train():
                self.train_dune(train_kwargs)

                if self.ask_to_continue():
                    self.model.load_state_dict(torch.load(self.full_model_name, map_location=torch.device('cpu')))
                    to_device(self.model)
                    self.model.eval()
                else:
                    print('You can set the new model path to the DUNE class to use the trained model.') 

            else:
                print('Can not find checkpoint. Please check the path or train first.')
                raise FileNotFoundError


    def train_dune(self, train_kwargs):

        model_name = train_kwargs.get("model_name", self.robot.name)

        checkpoint_path = sys.path[0] + '/model' + '/' + model_name
        checkpoint_path = repeat_mk_dirs(checkpoint_path)
        
        self.train_model = DUNETrain(self.model, self.G, self.h, checkpoint_path)
        self.full_model_name = self.train_model.start(**train_kwargs)
        print('Complete Training. The model is saved in ' + self.full_model_name)

    def ask_to_train(self):
        
        while True:
            choice = input("Do not find the DUNE model; Do you want to train the model now, input Y or N:").upper()
            if choice == 'Y':
                return True
            elif choice == 'N':
                print('Please set the your model path for the DUNE layer.')
                sys.exit()
            else:
                print("Wrong input, Please input Y or N.")


    def ask_to_continue(self):
        
        while True:
            choice = input("Do you want to continue the case running, input Y or N:").upper()
            if choice == 'Y':
                return True
            elif choice == 'N':
                print('exit the case running.')
                sys.exit()
            else:
                print("Wrong input, Please input Y or N.")


    @property
    def points(self):
        '''
        point considered in the dune layer
        '''

        return self.obstacle_points

