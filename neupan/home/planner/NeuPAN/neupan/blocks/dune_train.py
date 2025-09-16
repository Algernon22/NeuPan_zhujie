"""
DUNETrain is the class for training the DUNE model. It is used when you deploy the NeuPan algorithm on a new robot with a specific geometry. 

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
from colorama import deinit

deinit()

from torch.utils.data import Dataset, random_split, DataLoader
import cvxpy as cp   # 凸优化库, 用于构建凸优化问题
from rich.console import Console
from rich.progress import Progress
from rich.live import Live
from torch.optim import Adam
import numpy as np
from neupan.configuration import np_to_tensor, value_to_tensor, to_device
import pickle
import time
import os


class PointDataset(Dataset):   # 继承自torch.utils.data.Dataset, 用于自定义数据集
    def __init__(self, input_data, label_data, distance_data):
        """
        input_data: point p, [2, 1]         # 2D点坐标
        label_data: mu, [G.shape[0], 1]     # 每个约束边的权重
        distance_data: distance, scalar     # 点到约束边界的距离

        # 例如，一个矩形机器人的G矩阵可能是这样的：
        G = np.array([
            [ 1,  0],  # 右边界约束
            [-1,  0],  # 左边界约束
            [ 0,  1],  # 上边界约束
            [ 0, -1]   # 下边界约束
        ])
        G.shape = (4, 2)
        G.shape[0] = 4：表示有4个约束边
        G.shape[1] = 2：表示2D空间
        """

        self.input_data = input_data
        self.label_data = label_data
        self.distance_data = distance_data

    def __len__(self):
        return len(self.input_data)

    # 获取单个样本，用于DataLoader批量加载
    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        label_sample = self.label_data[idx]
        distance_sample = self.distance_data[idx]

        return input_sample, label_sample, distance_sample


class DUNETrain:
    def __init__(self, model, robot_G, robot_h, checkpoint_path) -> None:

        self.G = robot_G
        self.h = robot_h
        self.model = model   # ObsPointNet模型实例

        self.construct_problem()                 # 构造优化问题
        self.checkpoint_path = checkpoint_path   # 模型保存路径

        self.loss_fn = torch.nn.MSELoss()   # 均方误差损失函数
        self.optimizer = Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        # for rich progress 进度显示相关
        self.console = Console()
        self.progress = Progress(transient=False)
        self.live = Live(self.progress, console=self.console, auto_refresh=False)

        # loss 损失记录
        self.loss_of_epoch = 0
        self.loss_list = []

    def construct_problem(self):
        """
        optimization problem (10):

        max mu^T * (G * p - h)
        s.t. ||G^T * mu|| <= 1
             mu >= 0
        """
        self.mu = cp.Variable((self.G.shape[0], 1), nonneg=True)   # 定义优化变量μ,[行数 = G 的行数，列数 = 1], nonneg=True: 所有元素非负
        self.p = cp.Parameter((2, 1))  # points

        cost = self.mu.T @ (self.G.cpu() @ self.p - self.h.cpu())   # 构建目标函数
        constraints = [cp.norm(self.G.cpu().T @ self.mu) <= 1]      # 构建约束条件 ||Gᵀ·μ||₂ ≤ 1

        self.prob = cp.Problem(cp.Maximize(cost), constraints)   # 构建优化问题

    def process_data(self, rand_p):
        """
        处理随机点数据，求解优化问题并返回结果张量
        """
        distance_value, mu_value = self.prob_solve(rand_p)  # Adapted to be accessible
        return (
            np_to_tensor(rand_p),
            np_to_tensor(mu_value),
            value_to_tensor(distance_value),
        )

    def generate_data_set(self, data_size=10000, data_range=[-50, -50, 50, 50]):
        """
        创建数据集, 这个数据集里包含了点坐标, 以及该点对应的最优mu和最优距离
        generate dataset for training
        data_range: [low_x, low_y, high_x, high_y]
        """
        input_data = []
        label_data = []
        distance_data = []

        # 生成随机点坐标
        rand_p = np.random.uniform(
            low=data_range[:2],    # X和Y的最小值 [-50, -50]
            high=data_range[2:],   # X和Y的最大值 [50, 50]
            size=(data_size, 2)    # 生成data_size个二维点
        )
        # 将每个点重塑为 (2, 1) 形状 原始：[1.2, 3.4] → 重塑后：[[1.2], [3.4]]
        rand_p_list = [rand_p[i].reshape(2, 1) for i in range(data_size)]

        for p in rand_p_list:
            results = self.process_data(p)
            input_data.append(results[0])
            label_data.append(results[1])   # 对于一个随机点，mu不是单个值，而是一个权重向量，编码了该点与问题空间中所有相关边的完整关系
            distance_data.append(results[2])

        dataset = PointDataset(input_data, label_data, distance_data)

        return dataset

    def prob_solve(self, p_value):

        self.p.value = p_value
        self.prob.solve(solver=cp.ECOS)  # distance
        # self.prob.solve()  # distance

        # self.prob.value 目标函数的最优值, 点 p 到多边形的有符号距离     > 0：点在多边形外部; = 0：点在多边形边界或内部; < 0：不可能出现（因 mu > 0）
        # self.mu.value 优化变量的最优值，即每个约束的权重
        return self.prob.value, self.mu.value

    def start(
        self,
        data_size: int = 100000,
        data_range: list[int] = [-25, -25, 25, 25],
        batch_size: int = 256,
        epoch: int = 5000,
        valid_freq: int = 100,                         # 验证频率（每多少轮验证一次）
        save_freq: int = 500,                          # 模型保存频率（每多少轮保存一次）
        lr: float = 5e-5,
        lr_decay: float = 0.5,                         # 学习率衰减系数
        decay_freq: int = 1500,                        # 学习率衰减频率（每多少轮衰减一次）
        save_loss: bool = False,
        **kwargs,
    ):
        # 创建训练配置字典
        train_dict = {
            "data_size": data_size,
            "data_range": data_range,
            "batch_size": batch_size,
            "epoch": epoch,
            "valid_freq": valid_freq,
            "save_freq": save_freq,
            "lr": lr,
            "lr_decay": lr_decay,
            "decay_freq": decay_freq,
            "robot_G": self.G,
            "robot_h": self.h,
            "model": self.model,
        }

        # 保存配置到文件
        with open(self.checkpoint_path + "/train_dict.pkl", "wb") as f:
            pickle.dump(train_dict, f)

        # 打印配置并记录到日志文件
        print(
            f"data_size: {data_size}, data_range: {data_range}, batch_size: {batch_size}, epoch: {epoch}, valid_freq: {valid_freq}, save_freq: {save_freq}, lr: {lr}, lr_decay: {lr_decay}, decay_freq: {decay_freq}, robot_G: {self.G}, robot_h: {self.h}"
        )
        with open(self.checkpoint_path + "/results.txt", "a") as f:
            print(
                f"data_size: {data_size}, data_range: {data_range}, batch_size: {batch_size}, epoch: {epoch}, valid_freq: {valid_freq}, save_freq: {save_freq}, lr: {lr}, lr_decay: {lr_decay}, decay_freq: {decay_freq}, robot_G: {self.G}, robot_h: {self.h}\n",
                file=f,
            )

        self.optimizer.param_groups[0]["lr"] = float(lr)
        ful_model_name = None

        print("dataset generating start ...")
        dataset = self.generate_data_set(data_size, data_range)
        # 分割数据集（80%训练，20%验证）
        train, valid, _ = random_split(
            dataset, [int(data_size * 0.8), int(data_size * 0.2), 0]
        )
        # 创建数据加载器, 从dataset中用__getitem__方法取batch_size个数据
        train_dataloader = DataLoader(train, batch_size=batch_size)
        valid_dataloader = DataLoader(valid, batch_size=batch_size)

        print("dataset training start ...")

        # 使用rich库创建进度条
        with self.live:
            task = self.progress.add_task("[cyan]Training...", total=epoch)

            # 迭代训练
            for i in range(epoch + 1):
                self.progress.update(task, advance=1)   # 更新进度条
                self.live.refresh()

                self.model.train(True)   # 设置模型为训练模式

                # 训练一个epoch
                mu_loss, distance_loss, fa_loss, fb_loss = self.train_one_epoch(
                    train_dataloader, False
                )

                # 格式化损失值
                ml, dl, al, bl = (
                    "{:.2e}".format(mu_loss),
                    "{:.2e}".format(distance_loss),
                    "{:.2e}".format(fa_loss),
                    "{:.2e}".format(fb_loss),
                )

                # 模型验证
                if i % valid_freq == 0:
                    self.model.eval()   # 设置模型为评估模式
                    (
                        valid_mu_loss,
                        valid_distance_loss,
                        validate_fa_loss,
                        validate_fb_loss,
                    ) = self.train_one_epoch(valid_dataloader, True)

                    vml, vdl, val, vbl = (
                        "{:.2e}".format(valid_mu_loss),
                        "{:.2e}".format(valid_distance_loss),
                        "{:.2e}".format(validate_fa_loss),
                        "{:.2e}".format(validate_fb_loss),
                    )

                    self.print_loss(
                        i,
                        epoch,
                        ml,
                        dl,
                        al,
                        bl,
                        vml,
                        vdl,
                        val,
                        vbl,
                        self.optimizer.param_groups[0]["lr"],
                    )

                    with open(self.checkpoint_path + "/results.txt", "a") as f:
                        self.print_loss(
                            i,
                            epoch,
                            ml,
                            dl,
                            al,
                            bl,
                            vml,
                            vdl,
                            val,
                            vbl,
                            self.optimizer.param_groups[0]["lr"],
                            f,
                        )
                # 模型保存
                if i % save_freq == 0:
                    print("save model at epoch {}".format(i))
                    torch.save(
                        self.model.state_dict(),
                        self.checkpoint_path + "/" + "model_" + str(i) + ".pth",
                    )
                    ful_model_name = (
                        self.checkpoint_path + "/" + "model_" + str(i) + ".pth"
                    )

                # 学习率衰减
                if (i + 1) % decay_freq == 0:
                    self.optimizer.param_groups[0]["lr"] = (
                        self.optimizer.param_groups[0]["lr"] * lr_decay
                    )
                    print(
                        "current learning rate:", self.optimizer.param_groups[0]["lr"]
                    )

                    with open(self.checkpoint_path + "/results.txt", "a") as f:
                        print(
                            "current learning rate:",
                            self.optimizer.param_groups[0]["lr"],
                            file=f,
                        )

                self.loss_of_epoch = mu_loss + distance_loss + fa_loss + fb_loss   # 记录每个epoch的损失
                self.loss_list.append(self.loss_of_epoch)

                if save_loss:
                    with open(self.checkpoint_path + "/loss.pkl", "wb") as f:
                        pickle.dump(self.loss_list, f)

        print("finish train, the model is saved in {}".format(ful_model_name))

        return ful_model_name

    def train_one_epoch(self, train_dataloader, validate=False):
        """
        loss:
            mu: mse between output mu and label mu  预测μ与标签μ的MSE
            objective function value (distance): mse between output distance and label distance 预测距离与标签距离的MSE
            fa: -mu^T * G * R^T  ==> lam^T  λ向量的一致性
            fb: mu^T * G * R^T * p - mu^T * h  ==> lam^T * p + mu^T * h  目标函数值的一致性
        """

        mu_loss, distance_loss, fa_loss, fb_loss = 0, 0, 0, 0   # 为四种损失类型分别创建累加器，用于在整个epoch中累积损失值

        for input_point, label_mu, label_distance in train_dataloader:
            # input_point: 形状 [batch_size, 2]  一批点的坐标
            # label_mu: 形状 [batch_size, G.shape[0], 1]  每个点对应的最优μ值
            # label_distance: 形状 [batch_size]  每个点对应的最优距离值
            
            self.optimizer.zero_grad()   # 清零梯度

            input_point = torch.squeeze(input_point)    # 展平输入点，去除冗余维度，确保其形状为 (2,)
            output_mu = self.model(input_point)         # 输入模型获取预测的μ值
            output_mu = torch.unsqueeze(output_mu, 2)   # 展平输出mu，确保其形状为 (2, 1)

            distance = self.cal_distance(output_mu, input_point)   # 计算预测距离

            mse_mu = self.loss_fn(output_mu, label_mu)                              # 计算预测μ与标签μ的MSE
            mse_distance = self.loss_fn(distance, label_distance)                   # 计算预测距离与标签距离的MSE
            mse_fa, mse_fb = self.cal_loss_fab(output_mu, label_mu, input_point)    # 计算λ向量的一致性和目标函数值的一致性

            loss = mse_mu + mse_distance + mse_fa + mse_fb

            # 反向传播与优化
            if not validate:
                loss.backward()
                self.optimizer.step()

            # 累积损失值
            mu_loss += mse_mu.item()
            distance_loss += mse_distance.item()
            fa_loss += mse_fa.item()
            fb_loss += mse_fb.item()

        # 返回每个损失类型的平均值
        return (
            mu_loss / len(train_dataloader),
            distance_loss / len(train_dataloader),
            fa_loss / len(train_dataloader),
            fb_loss / len(train_dataloader),
        )

    def cal_loss_fab(self, output_mu, label_mu, input_point):
        """
        calculate the loss of fa and fb

        fa: -mu^T * G * R^T  ==> lam^T
        fb: mu^T * G * R^T * p - mu^T * h  ==> lam^T * p + mu^T * h
        """

        mu1 = output_mu     # 模型预测的μ [batch_size, 4]
        mu2 = label_mu      # 真实μ标签 [batch_size, 4]
        ip = torch.unsqueeze(input_point, 2)   # 输入点 [batch_size, 2, 1]
        mu1T = torch.transpose(mu1, 1, 2)      # 转置 [batch_size, 1, 4]
        mu2T = torch.transpose(mu2, 1, 2)      # 转置 [batch_size, 1, 4]

        # 生成随机旋转矩阵
        theta = np.random.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        R = np_to_tensor(R)

        fa = torch.transpose(-R @ self.G.T @ mu1, 1, 2)
        fa_label = torch.transpose(-R @ self.G.T @ mu2, 1, 2)

        fb = fa @ ip + mu1T @ self.h
        fb_label = fa_label @ ip + mu2T @ self.h

        mse_lamt = self.loss_fn(fa, fa_label)
        mse_lamtb = self.loss_fn(fb, fb_label)

        return mse_lamt, mse_lamtb

    def cal_distance(self, mu, input_point):

        input_point = torch.unsqueeze(input_point, 2)

        temp = self.G @ input_point - self.h

        muT = torch.transpose(mu, 1, 2)

        distance = torch.squeeze(torch.bmm(muT, temp))

        return distance

    def print_loss(self, i, epoch, ml, dl, al, bl, vml, vdl, val, vbl, lr, file=None):

        if file is None:
            print(
                "Epoch {}/{}, learning rate {} \n"
                "---------------------------------\n"
                "Losses:\n"
                "  Mu Loss:          {} | Validate Mu Loss:          {}\n"
                "  Distance Loss:    {} | Validate Distance Loss:    {}\n"
                "  Fa Loss:          {} | Validate Fa Loss:          {}\n"
                "  Fb Loss:          {} | Validate Fb Loss:          {}\n".format(
                    i,
                    epoch,
                    lr,
                    str(ml).ljust(10),
                    str(vml).rjust(10),
                    str(dl).ljust(10),
                    str(vdl).rjust(10),
                    str(al).ljust(10),
                    str(val).rjust(10),
                    str(bl).ljust(10),
                    str(vbl).rjust(10),
                )
            )

        else:
            print(
                "Epoch {}/{} learning rate {} \n"
                "---------------------------------\n"
                "Losses:\n"
                "  Mu Loss:          {} | Validate Mu Loss:          {}\n"
                "  Distance Loss:    {} | Validate Distance Loss:    {}\n"
                "  Fa Loss:          {} | Validate Fa Loss:          {}\n"
                "  Fb Loss:          {} | Validate Fb Loss:          {}\n".format(
                    i,
                    epoch,
                    lr,
                    str(ml).ljust(10),
                    str(vml).rjust(10),
                    str(dl).ljust(10),
                    str(vdl).rjust(10),
                    str(al).ljust(10),
                    str(val).rjust(10),
                    str(bl).ljust(10),
                    str(vbl).rjust(10),
                ),
                file=file,
            )

    def test(self, model_pth, train_dict_kwargs, data_size_list=0, **kwargs):

        with open(train_dict_kwargs, "rb") as f:
            train_dict = pickle.load(f)

        model = to_device(train_dict["model"])
        model.load_state_dict(torch.load(model_pth))
        data_range = train_dict["data_range"]

        print("dataset generating start ...")

        max_data_size = max(data_size_list)

        start_time = time.time()
        dataset = self.generate_data_set(max_data_size, data_range)
        data_generate_time = time.time() - start_time
        print(
            "data_size:", max_data_size, "dataset generating time: ", data_generate_time
        )

        # 遍历不同批次大小进行测试
        for data_size in data_size_list:
            test_dataloader = DataLoader(dataset, batch_size=data_size)

            mu_loss_list = []
            distance_loss_list = []
            fa_loss_list = []
            fb_loss_list = []
            inference_time_list = []

            for input_point, label_mu, label_distance in test_dataloader:
                average_loss_list, inference_time = self.test_one_epoch(
                    model, input_point, label_mu, label_distance, data_size
                )

                mu_loss_list.append(average_loss_list[0])
                distance_loss_list.append(average_loss_list[1])
                fa_loss_list.append(average_loss_list[2])
                fb_loss_list.append(average_loss_list[3])
                inference_time_list.append(inference_time)

            avg_mu_loss = sum(mu_loss_list) / len(mu_loss_list)
            avg_distance_loss = sum(distance_loss_list) / len(distance_loss_list)
            avg_fa_loss = sum(fa_loss_list) / len(fa_loss_list)
            avg_fb_loss = sum(fb_loss_list) / len(fb_loss_list)
            avg_inference_time = sum(inference_time_list) / len(inference_time_list)

            with open(os.path.dirname(model_pth) + "/test_results.txt", "a") as f:
                print(
                    "Model_name {}, Data_size {}, inference_time {} \n"
                    "---------------------------------\n"
                    "Losses:\n"
                    "  Mu Loss:          {} \n"
                    "  Distance Loss:    {} \n"
                    "  Fa Loss:          {} \n"
                    "  Fb Loss:          {} \n".format(
                        os.path.basename(model_pth),
                        data_size,
                        avg_inference_time,
                        str(avg_mu_loss).ljust(10),
                        str(avg_distance_loss).ljust(10),
                        str(avg_fa_loss).ljust(10),
                        str(avg_fb_loss).ljust(10),
                    ),
                    file=f,
                )

                # with open(os.path.dirname(model_pth) + '/results_dict.pkl', 'wb') as f:
                #     results_kwargs = { 'Model_name': os.path.basename(model_pth), 'Data_size': data_size, 'inference_time': sum(inference_time_list) / len(inference_time_list), 'mu_loss': sum(mu_loss_list)/ len(mu_loss_list), 'distance_loss': sum(distance_loss_list)/len(distance_loss_list), 'fa_loss': sum(fa_loss_list)/ len(fa_loss_list), 'fb_loss': sum(fb_loss_list) / len(fb_loss_list)}

                #     pickle.dump(results_kwargs, f)
        print(
            "finish test, the results are saved in {}".format(
                os.path.dirname(model_pth) + "/test_results.txt"
            )
        )

    def test_one_epoch(self, model, input_point, label_mu, label_distance, data_size):

        input_point = torch.squeeze(input_point)

        start_time = time.time()
        output_mu = model(input_point)
        inference_time = time.time() - start_time

        output_mu = torch.unsqueeze(output_mu, 2)

        distance = self.cal_distance(output_mu, input_point)

        mse_mu = self.loss_fn(output_mu, label_mu)
        mse_distance = self.loss_fn(distance, label_distance)
        mse_fa, mse_fb = self.cal_loss_fab(output_mu, label_mu, input_point)

        # loss = mse_mu.item() + mse_distance + mse_fa + mse_fb
        # average_loss_list = [mse_mu.item() / data_size, mse_distance.item() / data_size, mse_fa.item() / data_size, mse_fb.item() / data_size]

        loss_list = [mse_mu.item(), mse_distance.item(), mse_fa.item(), mse_fb.item()]

        # print('Data_size {}, inference_time {} \n'
        #             '---------------------------------\n'
        #             'Losses:\n'
        #             '  Mu Loss:          {} \n'
        #             '  Distance Loss:    {} \n'
        #             '  Fa Loss:          {} \n'
        #             '  Fb Loss:          {} \n'
        #             .format(data_size, inference_time,
        #                     str(average_loss_list[0]).ljust(10),
        #                     str(average_loss_list[1]).ljust(10),
        #                     str(average_loss_list[2]).ljust(10),
        #                     str(average_loss_list[3]).ljust(10)))
        return loss_list, inference_time
