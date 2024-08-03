import random
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, ConcatDataset, Subset
from torchvision import datasets, transforms
import numpy as np
import functools
from copy import deepcopy
import torch
import pandas as pd

class Location30(Dataset):
    def __init__(self, file_path, train=True, train_test_ratio=0.5, label_column=0, seed=42):
        # Set the seed for reproducibility
        np.random.seed(seed)
        
        self.data_frame = pd.read_csv(file_path, header=None)
        self.labels = torch.tensor(self.data_frame[label_column].to_numpy(), dtype=torch.int64) - 1
        self.data_frame.drop(label_column, inplace=True, axis=1)
        self.data = torch.tensor(self.data_frame.to_numpy(), dtype=torch.float)
        
        dataset_size = len(self.labels)
        split = int(np.floor(train_test_ratio * dataset_size))
        
        indices = list(range(dataset_size))
        np.random.shuffle(indices)  # Shuffle indices to ensure random split
        
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        if train:
            self.data_indices = train_indices
        else:
            self.data_indices = test_indices
    
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.data_indices[idx]
        image = self.data[actual_idx]
        label = self.labels[actual_idx]
        return image, label

class Purchase100(Dataset):
    def __init__(self, file_path, train=True, train_test_ratio=0.5, seed=42):
        # Set the seed for reproducibility
        np.random.seed(seed)
        
        data = np.load(file_path)
        self.data = torch.tensor(data['features'], dtype=torch.float)
        self.labels = torch.tensor(data['labels'], dtype=torch.int64)
        
        dataset_size = len(self.labels)
        split = int(np.floor(train_test_ratio * dataset_size))
        
        indices = list(range(dataset_size))
        np.random.shuffle(indices)  # Shuffle indices to ensure random split
        
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        if train:
            self.data_indices = train_indices
        else:
            self.data_indices = test_indices
    
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.data_indices[idx]
        image = self.data[actual_idx]
        label = self.labels[actual_idx]
        return image, label
    
def make_uniform_dataset_users(data, num_users=1, local_dataset_size=10000):
    data_loader = DataLoader(data, batch_size=local_dataset_size, shuffle=True)
    datasets = []
    for i, (x, y) in enumerate(data_loader):
        if i >= num_users: break
        user_data = TensorDataset(x, y)
        datasets.append(user_data)
    return datasets

def make_multiform_dataset_users(data, num_class, num_users=1, batch_size=64, bias=0.5):
    num_outputs = num_class
    num_workers = num_users
    
    bias_weight = bias
    other_group_size = (1-bias_weight) / (num_outputs-1)
    worker_per_group = num_workers / (num_outputs) # num_worker=nuser, num_ouputs=nclass
    
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)] 
    print(len(each_worker_data))
    
    train_data = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            upper_bound = (y.item()) * (1-bias_weight) / (num_outputs-1) + bias_weight # default=0.5
            lower_bound = (y.item()) * (1-bias_weight) / (num_outputs-1)
            rd = np.random.random_sample()
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size)+y.item()+1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.item()

            # assign a data point to a worker
            rd = np.random.random_sample()
            selected_worker = int(worker_group*worker_per_group + int(np.floor(rd*worker_per_group)))
            if (bias_weight == 0): selected_worker = np.random.randint(num_workers)
            each_worker_data[selected_worker].append(x)
            each_worker_label[selected_worker].append(y)

    # concatenate the data for each worker
    each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data] 
    each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]
    
    # random shuffle the workers
    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    
    # 将each_worker_data和each_worker_label合并为TensorDataset
    train_sets = [TensorDataset(each_worker_data[i], each_worker_label[i]) for i in range(num_workers)]

    return train_sets

def uniform_sample_from_user(datasets, num_member=500) -> TensorDataset:
    sample_per_user = num_member // len(datasets)
    rest_sample = num_member % len(datasets)
    # print(len(datasets), sample_per_user, rest_sample)
    sampled_x = []
    sampled_y = []
    for i, dataset in enumerate(datasets):
        if i < rest_sample:
            indices = torch.randperm(len(dataset))[:sample_per_user+1]
        else:
            indices = torch.randperm(len(dataset))[:sample_per_user]
        x, y = zip(*[dataset[i] for i in indices])
        sampled_x.append(torch.stack(x))
        sampled_y.append(torch.stack(y))
    
    sampled_x = torch.cat(sampled_x)
    sampled_y = torch.cat(sampled_y)
    return TensorDataset(sampled_x, sampled_y)

def get_attack_member_set(data, target_user_id=1, num_member=500):
    target_data = data[target_user_id-1]
    indices = torch.randperm(len(target_data))[:num_member]
    member_set = Subset(target_data, indices)
    return member_set

def get_cover_set(data, cover_set_sz):
    data_loader = DataLoader(data, shuffle=True, batch_size=cover_set_sz)
    for x, y in data_loader:
        cover_set = TensorDataset(x, y)
        break
    return cover_set

def get_attack_non_member_set(data, num):
    data_loader = DataLoader(data, shuffle=True, batch_size=num)
    for x, y in data_loader:
        member_set = TensorDataset(x, y)
        break
    return member_set

def Change2TensorDataset(concateDataset):
    all_data = []
    for dataset in concateDataset.datasets:
        # 每个 dataset 是 TensorDataset 类型
        data = list(dataset)
        all_data.extend(data)

    # 将数据列表拆分为输入和标签
    inputs, labels = zip(*all_data)

    # 创建一个新的 TensorDataset
    tensor_dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
    return tensor_dataset

def setup_data_without_attack(
    load_dataset_fn,
    num_users,
    size_local_ds,
    batch_size,
    size_testset,
    type_partition
):
    train_data, val_data, x_shape, num_class = load_dataset_fn()
    test_set = make_uniform_dataset_users(val_data, 1, size_testset)[0]
    if type_partition == 0:
        train_sets = make_uniform_dataset_users(train_data, num_users, size_local_ds)
    else:
        train_sets = make_multiform_dataset_users(train_data, num_class, num_users, batch_size)
        
    train_sets = [DataLoader(train_set, batch_size=batch_size) for train_set in train_sets]
    test_set = DataLoader(test_set, batch_size=batch_size)
    return train_sets, test_set, None, x_shape, num_class
    
def setup_data(
    load_dataset_fn,
    num_users,
    size_local_ds,
    batch_size,
    size_testset,
    type_partition, # 0 if iid, 1 non-iid
    num_member=500,
    num_non_member=500,
    num_cover=1000,
    setting='s1'
):
    train_data, val_data, x_shape, num_class = load_dataset_fn() # train_data已经是shuffle过的
    train_data, cover_data = random_split(train_data, [len(train_data) - num_cover, num_cover])
    cover_set = get_cover_set(cover_data, num_cover)

    test_set = make_uniform_dataset_users(val_data, 1, size_testset)[0]
    test_set = DataLoader(test_set, batch_size=batch_size)

    non_member_set = get_attack_non_member_set(val_data, num_non_member)
    
    if type_partition == 0:
        train_sets = make_uniform_dataset_users(train_data, num_users - 1, size_local_ds)        
    else:
        train_sets = make_multiform_dataset_users(train_data, num_class, num_users - 1, batch_size)        
        
        # # 统计每个train_set的label分布
        # for i in range(len(train_sets)):
        #     labels = {}
        #     for x, y in train_sets[i]:
        #         if int(y) in labels:
        #             labels[int(y)] += 1
        #         else:
        #             labels[int(y)] = 1
        #     print('user', i+1)
        #     for label, count in labels.items():
        #         print(f"Label {label}: {count} instances")
            
    if setting == 's1':
        # get member from user 1 by default
        mem_set = get_attack_member_set(train_sets, target_user_id=1, num_member=num_member)    
    elif setting == 's2':
        # get member from user 11(non-neigh)
        mem_set = get_attack_member_set(train_sets, target_user_id=11, num_member=num_member)
    elif setting == 's3':
        # get member from user 1 and user 20(both neigh)
        mem_set1 = get_attack_member_set(train_sets, target_user_id=1, num_member=num_member//2)
        mem_set2 = get_attack_member_set(train_sets, target_user_id=20, num_member=num_member-num_member//2)
        mem_set = Change2TensorDataset(ConcatDataset([mem_set1, mem_set2]))
    elif setting == 's4':
        # get member from user 11 and user 14(bother
        mem_set1 = get_attack_member_set(train_sets, target_user_id=11, num_member=num_member//2)
        mem_set2 = get_attack_member_set(train_sets, target_user_id=14, num_member=num_member-num_member//2)
        mem_set = Change2TensorDataset(ConcatDataset([mem_set1, mem_set2]))
    elif setting == 's5':
        # get member from user 1(neigh) and user 11(non-neigh)
        mem_set1 = get_attack_member_set(train_sets, target_user_id=1, num_member=num_member//2)
        mem_set2 = get_attack_member_set(train_sets, target_user_id=11, num_member=num_member-num_member//2)
        mem_set = Change2TensorDataset(ConcatDataset([mem_set1, mem_set2]))
    elif setting == 's6':
        # random sample 500 training sample from training data
        mem_set = uniform_sample_from_user(train_sets, num_member=num_member)
    else:
        raise Exception()
        
    train_sets = [DataLoader(train_set, batch_size=batch_size) for train_set in train_sets]
    attacker_set = ConcatDataset([mem_set, non_member_set])
    attacker_set = Change2TensorDataset(attacker_set)
    assert len(attacker_set) == num_member + num_non_member, f'num_mem={num_member},num_non_mem={num_non_member}, but actual attack set len={len(attacker_set)}'
    print(f'setting: {setting} attacker member set: {len(mem_set)}, non-member set: {len(non_member_set)}, cover set: {len(cover_set)}')
    assert len(train_sets) == num_users - 1, f'user train set allocate fail, len should be {num_users-1} but {len(train_sets)}'
    train_sets = [attacker_set] + train_sets
    assert len(train_sets) == num_users, f'attacker train set have not be added'

        
    return train_sets, test_set, cover_set, x_shape, num_class


class EarlyStopping:
    def __init__(self, patience):
        self.best = [None, -np.inf]
        self.patience = patience
        self.current_patience = patience

    def __call__(self, i, new):
        if new > self.best[1]:
            self.best = [i, new]
            self.current_patience = self.patience
            return False
        else:
            self.current_patience -= 1
            print(f"\t {i}--getting worse {self.best[1]} --> {new} patience->{self.current_patience}")
            if self.current_patience == 0:
                print(f"{i}--Early stop")
                return True
        return False
    
def setup_model(
    make_model,  # 例如 make_resnet20
    model_hparams,
    same_init,
):
    if same_init:
        model, loss_fn, optimizer, scheduler, metric_fn = make_model(*model_hparams)

        # 创建深拷贝模型的函数
        def copy_model():
            return deepcopy(model), loss_fn, optimizer, scheduler, metric_fn

        return copy_model
    else:
        return functools.partial(make_model, *model_hparams)

# Setting up the CIFAR-10 dataset
def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10(root='~/.torch', train=True, download=True, transform=transform)
    val_data = datasets.CIFAR10(root='~/.torch', train=False, download=True, transform=transform)
    train_size = len(train_data)
    val_size = len(val_data)
    num_class = 10
    x_shape = train_data[0][0].shape # (3, 32, 32)
    print(f'train size: {train_size}, val size: {val_size}')
    
    # for i in range(50):
    #     print(train_data[i][1])
    
    return train_data, val_data, x_shape, num_class

# load location30
def load_location30():
    path = r'./dataset/bangkok'
    num_class = 30 
    train_data = Location30(path, train=True, train_test_ratio=0.5, label_column=0, seed=42)
    val_data = Location30(path, train=False, train_test_ratio=0.5, label_column=0, seed=42)
    x_shape = train_data[0][0].shape
    
    return train_data, val_data, x_shape, num_class

def load_purchase100():
    path = r'./dataset/purchase100.npz'
    num_class = 100
    train_data = Purchase100(path, train=True, train_test_ratio=0.5, seed=42)
    val_data = Purchase100(path, train=False, train_test_ratio=0.5, seed=42)
    x_shape = train_data[0][0].shape
    
    return train_data, val_data, x_shape, num_class
    
    
def shuffle_labels(victim_set):
    """
    Shuffle labels in the given dataset.

    Args:
        victim_set (TensorDataset): The input dataset.

    Returns:
        DataLoader: The dataset with shuffled labels.
    """
    import torch
    inputs, labels = victim_set.tensors
    shuffled_labels = torch.randint(0, 10, labels.shape, dtype=torch.long)
    shuffled_dataset = TensorDataset(inputs, shuffled_labels)
    return DataLoader(shuffled_dataset, batch_size=50, shuffle=True)

def split_train_set(train_set):
        """
        Split the training dataset into member and non-member sets.

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
  

def get_random_coverset(cover_set, num_cover_sam=500):
        """
        Get a random subset of the cover dataset.

        Args:
            num_cover_sam (int): Number of samples in the cover subset.

        Returns:
            DataLoader: The cover subset.
        """
        cover_indices = torch.randperm(len(cover_set))[:num_cover_sam]
        inputs, labels = cover_set.tensors
        cur_cover_set = TensorDataset(inputs[cover_indices], labels[cover_indices])
        return DataLoader(cur_cover_set, batch_size=100, shuffle=True)  
    
# Usage
if __name__ == "__main__":
    # num_users = 10
    # size_local_ds = 1000
    # batch_size = 32
    # size_testset = 10000
    # type_partition = 0
    # num_members = 500
    # num_non_members = 500
    # num_cover = 1000


    # train_sets, test_set, cover_set, x_shape, num_class = setup_data(
    #     lambda: load_cifar10(),
    #     num_users,
    #     size_local_ds,
    #     batch_size,
    #     size_testset,
    #     type_partition,
    #     num_members,
    #     num_non_members,
    #     num_cover
    # )
    
    # print(f'attacker dataset size: {len(train_sets[0])}\nnormal user dataset size: {len(train_sets[1])}')
    # print(f'cover set size: {len(cover_set)}')
    
    # print(type(train_sets[0]), type(train_sets[1]))
    # member_set, non_member_set = split_train_set(train_sets[0])
    # attacker_train_set = shuffle_labels(train_sets[0])
    
    # print(len(attacker_train_set), len(member_set), len(non_member_set))
    
    # cur_cover_set = get_random_coverset(cover_set, 500)
    # print(f'cur cover loader size: {len(cur_cover_set)}')
    
    # print(f'x shape: {x_shape}')
    
    # print(f'user data loader len: {len(train_sets[1])}')
    # train_dataset, test_dataset, x_shape, num_class = load_location30()
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # print(f'train size: {len(train_loader)}, test size: {len(test_loader)} x shape: {x_shape}')
    # for images, labels in train_loader:
    #     print('Train batch:', images.shape, labels.shape)
    #     break
    
    train_dataset, test_dataset, x_shape, num_class = load_purchase100()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print(f'train size: {len(train_loader)}, test size: {len(test_loader)} x shape: {x_shape}')