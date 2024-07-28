import torch
from torch.utils.data import DataLoader, TensorDataset
from .user import *
from .attacker import *

def shuffle_labels(victim_set):
    """
    Shuffle labels in the given dataset.

    Args:
        victim_set (TensorDataset): The input dataset.

    Returns:
        DataLoader: The dataset with shuffled labels.
    """
    inputs, labels = victim_set.tensors
    shuffled_labels = torch.randint(0, 10, labels.shape, dtype=torch.long)
    shuffled_dataset = TensorDataset(inputs, shuffled_labels)
    return shuffled_dataset

  
class Agrevader_v2(Attacker):
    def __init__(self, name, make_model, train_set, cover_set, device, normal_train_iter, attack_type='norm', cover_try_time=5):
        """
        Initialize the Agrevader_v2 class.

        Args:
            name (str): The name of the attacker.
            make_model (function): A function to create the model.
            train_set (TensorDataset): The training dataset.
            cover_set (TensorDataset): The cover dataset.
        """
        print('setting up AgrEvader')
        self.name = name
        self.member_set, self.non_member_set = self.split_train_set(train_set)
        self.shuffle_train_set = DataLoader(shuffle_labels(train_set), batch_size=64)
        self.shuffle_train_set_iter = iter(self.shuffle_train_set)
        self.train_set = DataLoader(train_set, batch_size=500)
        self.train_set_iter = iter(self.train_set)
        self.cover_set = cover_set
        self.cover_set_loader = DataLoader(cover_set, batch_size=64)
        self.cover_set_iter = iter(self.cover_set_loader)

        self.neighbors = set()
        self.model, self.loss, self.opt_fn, self.sheduler_fn, self.metric = make_model()
        self.opt = self.opt_fn(self.model)
        # for param_group in self.opt.param_groups:
        #     param_group['lr'] = 0.01
        self.scheduler = self.sheduler_fn(self.opt)
        
        self.victim = self.neighbors
        
        self.attack_result = []
        self.result = None
        self.mem_result = None
        self.model_update_buffer = {}
        self.window_model_update_buffer = [None, None]
        self.window_local_model = [None, None]
        self.window_training_data = [None, None]
        self.window_gradient = [None, None]
        self.gradient = None
        self.history = []
        self.iter = 0
        self.byz = None
        self.w_victim = None
        self.attack_param = None
        self.w_cover = None
        
        self.device = device
        self.model = self.model.to(self.device)
        
        self.normal_train_iter = normal_train_iter
        self.cover_try_time = cover_try_time
        
        self.attack_type = attack_type
        print(f'attack type: {attack_type}')

    def split_train_set(self, train_set):
        """
        Split the training dataset into member and non-member sets. 
        the number of sample the two sets have is equal.

        Args:
            train_set (TensorDataset): The training dataset.

        Returns:
            Tuple[TensorDataset, TensorDataset]: The member and non-member sets.
        """
        inputs, labels = train_set.tensors
        mid = len(train_set) // 2
        member_set = TensorDataset(inputs[:mid], labels[:mid])
        non_member_set = TensorDataset(inputs[mid:], labels[mid:])
        return member_set, non_member_set

    def get_random_coverset(self, num_cover_sam=500):
        """
        Get a random subset of the cover dataset.

        Args:
            num_cover_sam (int): Number of samples in the cover subset.

        Returns:
            DataLoader: The cover subset.
        """
        cover_indices = torch.randperm(len(self.cover_set))[:num_cover_sam]
        inputs, labels = self.cover_set.tensors
        cur_cover_set = TensorDataset(inputs[cover_indices], labels[cover_indices])
        return DataLoader(cur_cover_set, batch_size=num_cover_sam, shuffle=True)

    def get_victim_w(self):
        """
        Train the model on the victim dataset and return the trained weights.

        Returns:
            List[torch.Tensor]: The trained weights of the model.
        """
        origin_param = [p.data.clone() for p in self.model.parameters()]
        self.model.train()
        # for x, y in self.shuffle_train_set:
        try:
            x, y = next(self.shuffle_train_set_iter) # user 一次训练一个batch也即（64个sample）
        except StopIteration:  # 当迭代器迭代完所有数据时重置
            self.shuffle_train_set_iter = iter(self.shuffle_train_set)
            x, y = next(self.shuffle_train_set_iter)
        x, y = x.to(self.device), y.to(self.device)
        self.opt.zero_grad()
        p, loss = self.compute_loss(x, y, self.model)
        loss = loss.mean()
        loss.backward()
        self.opt.step()
        self.scheduler.step()
        w_victim = [p.data.clone() for p in self.model.parameters()]
        for p, op in zip(self.model.parameters(), origin_param):
            p.data = op
        self.w_victim = w_victim
        return w_victim

    def get_cover_w(self, cur_cover_set):
        """
        Train the model on the cover dataset and return the trained weights.

        Args:
            cur_cover_set (DataLoader): The cover subset.

        Returns:
            List[torch.Tensor]: The trained weights of the model.
        """
        # origin_param = [p.data.clone() for p in self.model.parameters()]
        self.model.train()
        cur_cover_set_iter = iter(cur_cover_set)
        for i in range(len(cur_cover_set_iter)):
            x, y = next(cur_cover_set_iter)
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            p, loss = self.compute_loss(x, y, self.model, training=True)
            loss = loss.mean()
            loss.backward()
            self.opt.step()
        w_cover = [p.data.clone() for p in self.model.parameters()]

        self.w_cover = w_cover
        return w_cover

    def get_cover_w_iter(self):
        """
        Train the model on the cover dataset and return the trained weights.

        Args:
            cur_cover_set (DataLoader): The cover subset.

        Returns:
            List[torch.Tensor]: The trained weights of the model.
        """
        self.model.train()
        try:
            x, y = next(self.cover_set_iter) # user 一次训练一个batch也即（64个sample）
        except StopIteration:  # 当迭代器迭代完所有数据时重置
            self.cover_set_iter = iter(self.cover_set_loader)
            x, y = next(self.cover_set_iter)
        x, y = x.to(self.device), y.to(self.device)
        self.opt.zero_grad()
        p, loss = self.compute_loss(x, y, self.model, training=True)
        loss = loss.mean()
        loss.backward()
        self.opt.step()
        w_cover = [p.data.clone() for p in self.model.parameters()]

        self.w_cover = w_cover
        return w_cover
    
    def combine_vic_cov(self, w_victim, w_cover, ratio=0.7):
        """
        Combine the victim and cover weights.

        Args:
            w_victim (List[torch.Tensor]): The victim weights.
            w_cover (List[torch.Tensor]): The cover weights.
            ratio (float): the weight of w_victim 

        Returns:
            List[torch.Tensor]: The combined weights.
        """
        # com_param = agg_sum(w_victim, w_cover)
        # com_param = agg_div(com_param, 2)
        com_param = agg_mul(agg_sum(agg_mul(w_victim, ratio), agg_mul(w_cover, 1-ratio)), 1.0)
        return com_param

    def get_max_neigh_norm_diff(self):
        max_neigh_norm_diff = 0.0
        neigh_params_list = []
        for neigh_name, params in self.model_update_buffer.items():
            params = flat_tensor_list(params)
            params = torch.tensor(params)
            neigh_params_list.append(params)
        
        for i in range(len(neigh_params_list)):
            for j in range(i + 1, len(neigh_params_list)):
                norm_diff = torch.norm(neigh_params_list[i] - neigh_params_list[j])
                max_neigh_norm_diff = max(max_neigh_norm_diff, norm_diff)
        return max_neigh_norm_diff

    def get_max_neigh_unitnorm_diff(self):
        max_neigh_unitnorm_diff = 0.0
        neigh_params_list = []
        for neigh_name, params in self.model_update_buffer.items():
            params = flat_tensor_list(params)
            params = torch.tensor(params)
            neigh_params_list.append(params)
        
        for i in range(len(neigh_params_list)):
            for j in range(i + 1, len(neigh_params_list)):
                unit_param_i = neigh_params_list[i] / torch.norm(neigh_params_list[i])
                unit_param_j = neigh_params_list[j] / torch.norm(neigh_params_list[j])
                unitnorm_diff = torch.norm(unit_param_i - unit_param_j)
                max_neigh_unitnorm_diff = max(max_neigh_unitnorm_diff, unitnorm_diff)
        return max_neigh_unitnorm_diff

    def get_angle(self, param_i, param_j):
        if isinstance(param_i, np.ndarray):
            param_i = torch.from_numpy(param_i)
        if isinstance(param_j, np.ndarray):
            param_j = torch.from_numpy(param_j)
        
        dot_product = torch.dot(param_i, param_j)
        norm_i = torch.norm(param_i)
        norm_j = torch.norm(param_j)
        cos_theta = dot_product / (norm_i * norm_j)
        theta = torch.acos(cos_theta)
        theta_degrees = torch.rad2deg(theta)
        
        return theta_degrees.item()

    def get_max_neigh_angle_diff(self):
        max_neigh_angle_diff = 0.0
        neigh_params_list = []
        for neigh_name, params in self.model_update_buffer.items():
            params = flat_tensor_list(params)
            params = torch.tensor(params)
            neigh_params_list.append(params)
        
        for i in range(len(neigh_params_list)):
            for j in range(i + 1, len(neigh_params_list)):
                theta_degrees = self.get_angle(neigh_params_list[i], neigh_params_list[j])
                max_neigh_angle_diff = max(max_neigh_angle_diff, theta_degrees)
        return max_neigh_angle_diff

    def get_best_attack_params(self, w_victim):
        """
        Get the best attack parameters by combining victim and cover weights.

        Args:
            w_victim (List[torch.Tensor]): The victim weights.

        Returns:
            List[torch.Tensor]: The best attack parameters.
        """
        best_attack_params = None
        # times = self.cover_try_time
        times = 1
        
        if self.attack_type == 'norm':
            max_neigh_diff = self.get_max_neigh_norm_diff()
        elif self.attack_type == 'unitnorm':
            max_neigh_diff = self.get_max_neigh_unitnorm_diff()
        elif self.attack_type == 'angle':
            max_neigh_diff = self.get_max_neigh_angle_diff()
        else: 
            raise Exception()
        
        while times:
            times -= 1
            # cur_cover_set = self.get_random_coverset()
            # w_cur_cover = self.get_cover_w(cur_cover_set)
            w_cur_cover = self.get_cover_w_iter()
            cur_attack_param = self.combine_vic_cov(w_victim, w_cur_cover)
            # cur_attack_param_flat = torch.cat([p.view(-1) for p in cur_attack_param])
            # neigh_params_list = [torch.cat([p.view(-1) for p in params]) for params in self.model_update_buffer.values()]
            cur_attack_param_flat = flat_tensor_list(cur_attack_param)
            neigh_params_list = [flat_tensor_list(params) for params in self.model_update_buffer.values()]
            
            # transfer to tensor
            neigh_params_list = [torch.from_numpy(neigh_param) for neigh_param in neigh_params_list]
            cur_attack_param_flat = torch.from_numpy(cur_attack_param_flat)
            
            # norm 
            if self.attack_type == 'norm':
                max_attacker_neigh_diff = max(torch.norm(neigh_param - cur_attack_param_flat) for neigh_param in neigh_params_list)
            # unit norm
            elif self.attack_type == 'unitnorm':
                max_attacker_neigh_diff = max(torch.norm(neigh_param / torch.norm(neigh_param) - cur_attack_param_flat / torch.norm(cur_attack_param_flat)) for neigh_param in neigh_params_list)
            # angle
            elif self.attack_type == 'angle':
                max_attacker_neigh_diff = max(self.get_angle(neigh_param, cur_attack_param_flat) for neigh_param in neigh_params_list)
            
            if max_attacker_neigh_diff < max_neigh_diff:
                if best_attack_params is None or torch.norm(torch.cat([p.view(-1) for p in best_attack_params])) < torch.norm(torch.cat([p.view(-1) for p in cur_attack_param])):
                    best_attack_params = cur_attack_param

        if best_attack_params is None:
            print('have not find best attack params!')
            # return [(p / torch.norm(p.view(-1))) for p in cur_attack_param]
            return cur_attack_param
        else:
            # print(f'find best attack params!, neigh diff: {max_neigh_diff}, attacker neigh diff: {max_attacker_neigh_diff}')
            # return [(p / torch.norm(p.view(-1))) for p in best_attack_params]
            return best_attack_params
    
    def evaluate_attack_result(self):
        """
        Evaluate the attack result and save it to a CSV file.
        """
        true_member_acc, false_member_acc, true_nonmember_acc, false_nonmember_acc = [], [], [], []
        true_member, false_member, true_nonmember, false_nonmember = 0, 0, 0, 0

        self.model.eval()
        with torch.no_grad():
            for x, y in DataLoader(self.non_member_set, batch_size=1):
                x, y = x.to(self.device), y.to(self.device)
                p = self.model(x)
                metric = self.metric(y, p)
                false_member_acc.append(metric.item())
                true_nonmember_acc.append(1 - metric.item())
                false_member += metric.item()
                true_nonmember += (1 - metric.item())

            for x, y in DataLoader(self.member_set, batch_size=1):
                x, y = x.to(self.device), y.to(self.device)
                p = self.model(x)
                metric = self.metric(y, p)
                true_member_acc.append(metric.item())
                false_nonmember_acc.append(1 - metric.item())
                true_member += metric.item()
                false_nonmember += (1 - metric.item())

        attack_accuracy = (true_member + true_nonmember) / (true_member + true_nonmember + false_member + false_nonmember)
        attack_precision = true_member / (true_member + false_member)
        attack_recall = true_member / (true_member + false_nonmember)
        self.result = (attack_accuracy, attack_precision, attack_recall)
        self.mem_result = (int(true_member), int(false_member), int(true_nonmember), int(false_nonmember))

    def train(self):
        """
        Train the model and perform the attack.
        """

        # # test only use normal train set(without label shuffling)
        # self.model.train()
        # for x, y in self.train_set:
        #     x, y = x.to(self.device), y.to(self.device)
        #     self.opt.zero_grad()
        #     p, loss = self.compute_loss(x, y, self.model)
        #     loss = loss.mean()
        #     loss.backward()
        #     self.opt.step()
        # self.scheduler.step()
        
        # # get data
        # # for i in range(len(self.train_set_iter)):
        # try:
        #     x, y = next(self.train_set_iter) # user 一次训练一个batch也即（64个sample）
        # except StopIteration:  # 当迭代器迭代完所有数据时重置
        #     self.train_set_iter = iter(self.train_set)
        #     x, y = next(self.train_set_iter)

        # x, y = x.to(self.device), y.to(self.device)
        # self.opt.zero_grad()
        # p, loss = self.compute_loss(x, y, self.model, training=True)
        # loss = loss.mean()
        
        # metric = self.metric(y, p)
        
        # loss.backward()
        # self.opt.step()
        # self.scheduler.step()
        
        ''' attack try time = 1'''
        # w_victim = self.get_victim_w() # test acc 没问题
        # attack_param = self.get_best_attack_params(w_victim)
        # # # w_cur_cover = self.get_cover_w_iter()
        # # attack_param = self.combine_vic_cov(w_victim, w_cur_cover)
        # self.attack_param = attack_param
        
        ''' cover set bs=64 '''
        try:
            x, y = next(self.cover_set_iter) # user 一次训练一个batch也即（64个sample）
        except StopIteration:  # 当迭代器迭代完所有数据时重置
            self.cover_set_iter = iter(self.cover_set_loader)
            x, y = next(self.cover_set_iter)

        x, y = x.to(self.device), y.to(self.device)
        self.opt.zero_grad()
        p, loss = self.compute_loss(x, y, self.model, training=True)
        loss = loss.mean()
        
        metric = self.metric(y, p)
        
        loss.backward()
        self.opt.step()
        self.scheduler.step()


    # def get_model_update(self, epoch):
    #     """
    #     Get the model update for the given user.

    #     Args:
    #         user: The user requesting the model update.

    #     Returns:
    #         List[torch.Tensor]: The attack parameters.
    #     """
    #     if epoch < self.normal_train_iter:
    #         var = self.w_cover
    #     else:
    #         var = self.attack_param
    #     # var = self.w_cover
    #     var = clone_list_tensors(var)
    #     return var


    # def update(self):
    #     """ Update state based on received model updates (except the attacker model) """
        
    #     nups = len(self.model_update_buffer) - 1
    #     new_theta = [torch.zeros_like(param) for param in self.model.parameters()]

    #     num_neigh = 0
    #     for user_name, theta in self.model_update_buffer.items():
    #         if user_name == self.name:
    #             continue
    #         num_neigh += 1
    #         for i, param in enumerate(theta):
    #             new_theta[i] += param
    #     assert num_neigh == nups, f'aggregate error {num_neigh} vs. {nups}'
    #     new_theta = agg_div_param(new_theta, nups)
        
    #     # logging
    #     if self.window_model_update_buffer[1] is None:
    #         self.window_model_update_buffer[0] = None
    #     else:
    #         self.window_model_update_buffer[0] = self.window_model_update_buffer[1].copy()
    #     self.window_model_update_buffer[1] = self.model_update_buffer.copy()
        
    #     # set new params for local model
    #     for param, new_param in zip(self.model.parameters(), new_theta):
    #         param.data = new_param
        
    #     # logging
    #     self.window_local_model[0] = self.window_local_model[1]
    #     self.window_local_model[1] = [param.data.clone() for param in self.model.parameters()]
    #     self.iter += 1





