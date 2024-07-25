import numpy as np
import networkx as nx
import math, random

from .utils import *
from .user import *

class DecentralizedLearning:
    
    def __init__(
        self,
        graph_properties
    ):
        self.graph_properties = graph_properties
        self.attacker = None
        self.U = None
        self.test_set = None
        self.n_users = None
        
        self.global_model = None
        self.device = None
        
    def setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            cover_set,
            user,
            attacker,
            device,
            normal_train_iter,
            attack_type
    ):
        
        assert len(train_sets) == n_users
        
        self.device = device
        self.U = [None] * n_users #! 初始化一个长度为 n_users 的列表 self.U
        self.n_users = n_users
        self.test_set = test_set
        self.normal_train_iter = normal_train_iter
        
        # attacker is always the first user
        self.attacker = attacker(0, make_model, train_sets[0], cover_set, device, self.normal_train_iter, attack_type) #! 使用的是trainset[0] and cover set
        # self.attacker = attacker(0, make_model, train_sets[0],  device)
        self.U[0] = self.attacker
        
        for i in range(1, self.n_users):
            self.U[i] = user(i, make_model, train_sets[i], device)    

    #! 从 networkx 图 (G) 中获取通信拓扑
    def from_nx_graph(
        self,
        G,
        make_model,
        train_sets,
        test_set,
        cover_set,
        user,
        attacker,
        device,
        normal_train_iter,
        attack_type,
        shuffle=True,
    ):
        """ Comm. topology from networkx graph """
        
        nodes = list(G.nodes)
        DecentralizedLearning.setup(
            self,
            len(nodes),
            make_model,
            train_sets,
            test_set,
            cover_set,
            user,
            attacker,
            device,
            normal_train_iter,
            attack_type
        )
        
        mmap = {} #! 将 NetworkX 图 (G) 的节点映射到用户数组 self.U 的索引
        if shuffle:
            random.shuffle(nodes)
        
        for i in range(self.n_users):
            u = nodes[i]
            mmap[u] = i
                  
        for u, v in G.edges:
            u = mmap[u]
            v = mmap[v]
            self.U[u].neighbors.add(self.U[v]) #! 存放的是邻居的完整实例
            self.U[v].neighbors.add(self.U[u])      
           
    #! 一轮迭代过程
    def __call__(self, epoch): 
        """ Exucutes one round: local training + communication + aggregation for every user """
        
        # get model updates
        pool = self.U[1:] #! 从第二个用户开始 (第一个用户是攻击者)
        for u in pool:
            u.train()            
            mu = u.get_model_update()
            u.model_update_buffer[u.name] = mu
            u.check_model(u.name, mu)
            
            # send the local model update to neighbors
            for v in u.neighbors:
                mu = u.get_model_update() #! 拿到u模型更新θ(t+1/2), 发送给v
                v.check_model(u.name, mu)
                # 先存着
                v.model_update_buffer[u.name] = mu
                    
                    
        # attacker acts after everyone else (only for active attacks, when needed)
        self.attacker.train()  
        mu = self.attacker.get_model_update(epoch) #! mu是攻击者新的模型参数
        if epoch < self.normal_train_iter:
            self.attacker.model_update_buffer[self.attacker.name] = mu #! 保存攻击者的模型参数
        for v in self.attacker.neighbors:
            # 复制一份当前的参数出来
            mu = self.attacker.get_model_update(epoch)
            v.check_model(self.attacker.name, mu)
            v.model_update_buffer[self.attacker.name] = mu
        
                    
        pool = self.U
        for u in pool:
            u.check_models()
            # 正常client更新参数
            u.update()
    
       
    def compute_global_model(self, drop_attacker=True):
        """ Computes the global model i.e., the average of all local models (but the attacker on the malicious model)"""
        
        # init arch
        if self.global_model is None:
            self.global_model = deepCopyModel(self.attacker.model)
            self.global_model = self.global_model.to(self.device)
            
        # init weights to 0
        global_vars = init_list_variables(self.attacker.model)
        global_buffers = init_list_buffers(self.attacker.model)
            
        # do not consider the attaker when active
        if drop_attacker:
            users = self.U[1:]
        else:
            users = self.U
        
        # avg local models
        for u in users:
            u_params = [p.data.clone() for p in u.model.parameters()]
            u_buffer = [b.data.clone() for b in u.model.buffers()]
            global_vars = agg_sum(global_vars, u_params)
            global_buffers = agg_sum(global_buffers, u_buffer)
        global_vars = agg_div(global_vars, len(users))
        global_buffers = agg_div(global_buffers, len(users))
        
        # assign_list_variables(self.global_model.parameters(), global_vars)
        for p, op in zip(self.global_model.parameters(), global_vars):
            p.data = op.clone()
        for p, op in zip(self.global_model.buffers(), global_buffers):
            p.data = op.clone()
            
        return self.global_model
    
    
    def train_test_utility(self, drop_attacker=True):
        
        self.compute_global_model(drop_attacker=drop_attacker) # aggregate所有user的参数得到全局模型
        
        if drop_attacker:
            users = self.U[1:]
        else:
            users = self.U
        
        test_acc_lst = []
        loss_train, acc_train = 0., 0.
        for idx, u in enumerate(users):
            self_loss, self_acc = u.evaluate(self.test_set, model=u.model)[:2]
            test_acc_lst.append(self_acc)
            print(f'User {u.name} test acc: {self_acc}')
            # 把所有user的训练集在全局模型上跑一次得到train的acc
            _loss_train, _acc_train = u.evaluate(u.train_set, model=self.global_model)[:2]
            loss_train += _loss_train
            acc_train += _acc_train

        loss_train = loss_train / len(users)
        acc_train = acc_train / len(users)
        
        # 在attacker上跑一次得到test的acc（其实谁跑测试集都无所谓，都是调用全局模型）
        loss_test, acc_test = self.U[1].evaluate(self.test_set, model=self.global_model)[:2]

        
        return (loss_train, acc_train), (loss_test, acc_test), test_acc_lst
    
    
    def model_graph(self, with_labels=True, node_color=None):
       
        self.G = nx.Graph()
        for i in range(self.n_users):
            self.G.add_node(self.U[i].name)
            
        for u in self.U:
            for v in u.neighbors: 
                e = (u.name, v.name) if u.name > v.name else (v.name, u.name)
                self.G.add_edge(*e)
                
        if node_color is None:
            node_color = [[0, 1, 0]] * self.n_users
            # attacker
            node_color[0] = [1, 0, 0]
            # attacker's neighbors
            for v in self.G.neighbors(self.attacker.name):
                node_color[int(v)] = [1, .7, .8]
                      
        nx.draw(self.G, with_labels=with_labels, node_color=node_color)
        
        
    def compute_models_distance(self):
        """ Computes the consensus distance """
        m = flat_tensor_list(self.U[0].model.parameters()).shape[0]
        
        params = np.zeros((self.n_users,m))
        for i, u in enumerate(self.U):
            params[i] = flat_tensor_list(u.model.parameters())

        n = params.shape[0]
        dmatrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dmatrix[i,j] = ((params[i] - params[j]) ** 2).mean()
        return dmatrix        
    
#Pre-defined top. -------------------------------------------------------------------------------------
    
# k正则图
class Regular15(DecentralizedLearning):
    def setup(
        self,
        n_users,
        make_model,
        train_sets,
        test_set,
        cover_set,
        user,
        attacker,
        device,
        normal_train_iter,
        attack_type
    ):
        G = nx.random_regular_graph(15, n_users, seed=0) #! 30_15: 0号的neighbor为 1, 4, 5, 7, 8, 10, 12, 15, 20, 21, 22, 23, 25, 26, 29
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, cover_set, user, attacker, device, normal_train_iter, attack_type, shuffle=False)


class Regular20(DecentralizedLearning):
    def setup(
        self,
        n_users,
        make_model,
        train_sets,
        test_set,
        cover_set,
        user,
        attacker,
        device,
        normal_train_iter,
        attack_type
    ):
        G = nx.random_regular_graph(20, n_users, seed=0) #! 40_20: 0号的neighbor为 1, 4, 6, 9, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 32, 33, 34, 36, 38
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, cover_set, user, attacker, device, normal_train_iter, attack_type, shuffle=False)
   
   
class Regular25(DecentralizedLearning):
    def setup(
        self,
        n_users,
        make_model,
        train_sets,
        test_set,
        cover_set,
        user,
        attacker,
        device,
        normal_train_iter,
        attack_type
    ):
        G = nx.random_regular_graph(25, n_users, seed=0) #! 50_25: 0号的neighbor为 1, 2, 3, 6, 7, 10, 12, 13, 15, 18, 19, 20, 23, 25, 28, 30, 31, 32, 33, 34, 35, 38, 39, 43, 44
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, cover_set, user, attacker, device, normal_train_iter, attack_type, shuffle=False)
             

# 环图
class Ring(DecentralizedLearning):
    def setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            cover_set,
            user,
            attacker,
            device,
            normal_train_iter
    ):
        
        G = nx.cycle_graph(n_users)
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, cover_set, user, attacker, device, shuffle=False)

                            

class Torus(DecentralizedLearning):
     def setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            cover_set,
            user,
            attacker,
            device,
            normal_train_iter
    ):
        
        """ Torus comm. topology. n_users must have a square root """
        n = math.sqrt(n_users)
        assert n % 1 == 0
        
        n = int(n)
        G = nx.grid_graph(dim =[n, n], periodic=True) #! 
        
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, cover_set, user, attacker, device, shuffle=False)
            

# 完全图
class Complete(DecentralizedLearning):
    def setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            cover_set,
            user,
            attacker,
            device,
            normal_train_iter,
            attack_type
    ):
        
        G = nx.complete_graph(n_users)
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, cover_set, user, attacker, device, normal_train_iter, attack_type, shuffle=False)
        

# 随机图
class Random(DecentralizedLearning):
    def setup(
            self,
            n_users,
            make_model,
            train_sets,
            test_set,
            cover_set,
            user,
            attacker,
            device,
            normal_train_iter
    ):
        
        G = nx.erdos_renyi_graph(n_users, 0.1, seed=0)
        DecentralizedLearning.from_nx_graph(self, G, make_model, train_sets, test_set, cover_set, user, attacker, device, shuffle=False)
        
