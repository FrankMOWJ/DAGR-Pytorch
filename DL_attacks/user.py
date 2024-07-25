import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .ops_on_vars_list import *

class User:
    def __init__(self, name, make_model, train_set, device):
        self.name = name  # 实际上就是user的编号
        self.train_set = train_set
        self.train_set_iter = iter(self.train_set)  # 使用iter对象
        self.neighbors = set()  # 存放的是邻居的完整实例
        self.model, self.loss, self.opt, self.sheduler, self.metric = make_model()
        self.opt = self.opt(self.model)  # 优化器
        self.sheduler = self.sheduler(self.opt) # 适配器
        
        # received model updates at the current round
        self.model_update_buffer = {}  # 保存所有邻居的参数
        
        # 保存最新两个
        self.window_model_update_buffer = [None, None]
        self.window_local_model = [None, None]
        self.window_training_data = [None, None]
        self.window_gradient = [None, None]
        self.gradient = None 
        
        # 保存训练历史
        self.history = []
        self.iter = 0
        
        # self.byz = 'trim'  # 鲁棒机制(None, 'trim', 'median')
        self.byz = None
        
        self.device = device
        self.model = self.model.to(device)
        
    def get_model_update(self, epoch=0): 
        """ Generate model update for user 'user' """ 
        return [param.data.clone() for param in self.model.parameters()]
    
    def compute_loss(self, x, y, model, training=True):
        model.train() if training else model.eval()
        p = model(x)
        loss = self.loss(p, y)
        return p, loss
    
    def train(self):
        """ Local training step """
        
        # get data
        try:
            x, y = next(self.train_set_iter)
        except StopIteration:  # 当迭代器迭代完所有数据时，手动重置
            self.train_set_iter = iter(self.train_set)
            x, y = next(self.train_set_iter)
        # x, y = next(self.train_set_iter) # user 一次训练一个batch也即（64个sample）

        x, y = x.to(self.device), y.to(self.device)
        # print(f'user {self.name} x shape: {x.shape}')
        self.opt.zero_grad()
        p, loss = self.compute_loss(x, y, self.model, training=True)
        loss = loss.mean()
        
        # print(f'pred shape: {p.shape}, y shape: {y.shape}')
        metric = self.metric(y, p)
        
        loss.backward()
        self.opt.step()
        self.sheduler.step()
        
        # logging 
        # 这里可能有点问题
        self.gradient = [param.grad.clone() for param in self.model.parameters()]

        out = (loss.item(), metric.item())
        self.history.append(out)
        
        self.window_gradient[0] = self.window_gradient[1]
        self.window_gradient[1] = self.gradient
        
        self.window_training_data[0] = self.window_training_data[1]
        self.window_training_data[1] = x.clone().detach().cpu().numpy()
    
    def trimmed_mean(self, cmax=0):
        assert (cmax <= len(self.model_update_buffer) // 2), "cmax should be less than half of the neighbors"
        
        # 获取邻居的更新参数
        updates = list(self.model_update_buffer.values())
        updates.append([param.data.clone() for param in self.model.parameters()])

        # 将更新参数列表转换为参数向量列表
        param_list = torch.stack([torch.cat([param.view(-1) for param in update]) for update in updates])

        # 对每一列进行排序
        sorted_array = torch.sort(param_list, dim=0)[0]

        # 计算修剪均值
        n = param_list.shape[0]
        trimmed_mean = torch.mean(sorted_array[cmax:n-cmax, :], dim=0)

        # 将修剪均值向量分配回模型参数
        new_theta = []
        idx = 0
        for param in self.model.parameters():
            num_elements = param.numel()
            new_param = trimmed_mean[idx:idx+num_elements].view(param.size())
            new_theta.append(new_param)
            idx += num_elements

        return new_theta
    
    def median(self):
        # 获取邻居的更新参数
        updates = list(self.model_update_buffer.values())
        updates.append([param.data.clone() for param in self.model.parameters()])

        # 将更新参数列表转换为参数向量列表
        param_list = torch.stack([torch.cat([param.view(-1) for param in update]) for update in updates])

        # 对每一列进行排序
        sorted_array = torch.sort(param_list, dim=0)[0]

        # 计算中位数
        n = param_list.shape[0]
        if n % 2 == 1:
            median = sorted_array[n // 2, :]
        else:
            median = (sorted_array[n // 2 - 1, :] + sorted_array[n // 2, :]) / 2

        # 将中位数向量分配回模型参数
        new_theta = []
        idx = 0
        for param in self.model.parameters():
            num_elements = param.numel()
            new_param = median[idx:idx+num_elements].view(param.size())
            new_theta.append(new_param)
            idx += num_elements

        return new_theta
    
    def update(self):
        """ Update state based on received model updates (and local) """
        
        nups = len(self.model_update_buffer)
        new_theta = [torch.zeros_like(param) for param in self.model.parameters()]

        if self.byz is None:
            for theta in self.model_update_buffer.values():
                for i, param in enumerate(theta):
                    new_theta[i] += param
            new_theta = agg_div_param(new_theta, nups)
        elif self.byz == 'trim':
            new_theta = self.trimmed_mean()
        elif self.byz == 'median':
            new_theta = self.median()
        
        # logging
        if self.window_model_update_buffer[1] is None:
            self.window_model_update_buffer[0] = None
        else:
            self.window_model_update_buffer[0] = self.window_model_update_buffer[1].copy()
        self.window_model_update_buffer[1] = self.model_update_buffer.copy()
        
        # set new params for local model
        for param, new_param in zip(self.model.parameters(), new_theta):
            param.data = new_param
        
        # logging
        self.window_local_model[0] = self.window_local_model[1]
        self.window_local_model[1] = [param.data.clone() for param in self.model.parameters()]
        self.iter += 1

    def check_model(self, user, var):
        """ hook function: called when the user receive the model update from user ‘user’ """
        pass
        
    def check_models(self):
        """ hook function: called when all the model update have been received (before update()) """
        pass

    def evaluate(self, dataset, model=None):
        loss = []
        metric = []
        output = []
        Y = []
        
        if model is None:
            model = self.model
        
        model.eval()
        tot_loss = 0.0
        tot_acc = 0.0
        with torch.no_grad():
            for x, y in dataset:
                x = x.to(self.device)
                y = y.to(self.device)
                p, _loss = self.compute_loss(x, y, model, training=False)
                _metric = self.metric(y, p)
                
                # loss.append(_loss.cpu().numpy())
                # metric.append(_metric.cpu().numpy())
                # output.append(p.cpu().numpy())
                # Y.append(y.cpu().numpy())
                tot_loss += _loss.item()
                tot_acc += _metric.item()
               
        # loss = np.concatenate(loss)
        # metric = np.concatenate(metric)
        # output = np.concatenate(output)
        # Y = np.concatenate(Y)
        
        return tot_loss / len(dataset), tot_acc / len(dataset)
      
    def __repr__(self):
        return "User: " + str(self.name)
