from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, ConcatDataset, Subset
from torchvision import datasets, transforms
import numpy as np
import functools
from copy import deepcopy
import torch

def make_uniform_dataset_users(data, num_users=1, local_dataset_size=10000) -> list[TensorDataset]:
    data_loader = DataLoader(data, batch_size=local_dataset_size, shuffle=True)
    datasets = []
    for i, (x, y) in enumerate(data_loader):
        if i >= num_users: break
        user_data = TensorDataset(x, y)
        datasets.append(user_data)
    return datasets

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
    target_data = data[target_user_id]
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
    train_sets = [DataLoader(train_set, batch_size=batch_size) for train_set in train_sets]
    test_set = DataLoader(test_set, batch_size=batch_size)
    return train_sets, test_set, None, x_shape, num_class
    
def setup_data(
    load_dataset_fn,
    num_users,
    size_local_ds,
    batch_size,
    size_testset,
    type_partition,
    num_member=500,
    num_non_member=500,
    num_cover=1000,
    setting='s1'
):
    train_data, val_data, x_shape, num_class = load_dataset_fn()

    train_data, cover_data = random_split(train_data, [len(train_data) - num_cover, num_cover])
    cover_set = get_cover_set(cover_data, num_cover)

    test_set = make_uniform_dataset_users(val_data, 1, size_testset)[0]
    test_set = DataLoader(test_set, batch_size=batch_size)

    non_member_set = get_attack_non_member_set(val_data, num_non_member)
    
    if type_partition == 0:
        train_sets = make_uniform_dataset_users(train_data, num_users - 1, size_local_ds)
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
            # get member from user 11 and user 14(bother non-neigh)
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
        
        # cover_set = train_sets[0]
        # train_sets = train_sets[1:]
        train_sets = [DataLoader(train_set, batch_size=batch_size) for train_set in train_sets]
        attacker_set = ConcatDataset([mem_set, non_member_set])
        attacker_set = Change2TensorDataset(attacker_set)
        assert len(attacker_set) == num_member + num_non_member, f'num_mem={num_member},num_non_mem={num_non_member}, but actual attack set len={len(attacker_set)}'
        print(f'setting: {setting} attacker member set: {len(mem_set)}, non-member set: {len(non_member_set)}, cover set: {len(cover_set)}')
        assert len(train_sets) == num_users - 1, f'user train set allocate fail, len should be {num_users-1} but {len(train_sets)}'
        train_sets = [attacker_set] + train_sets
        assert len(train_sets) == num_users, f'attacker train set have not be added'
        
    else:
        raise Exception()

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
    x_shape = train_data[0][0].shape
    print(f'train size: {train_size}, val size: {val_size}')
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
    num_users = 10
    size_local_ds = 1000
    batch_size = 32
    size_testset = 10000
    type_partition = 0
    num_members = 500
    num_non_members = 500
    num_cover = 1000


    train_sets, test_set, cover_set, x_shape, num_class = setup_data(
        lambda: load_cifar10(),
        num_users,
        size_local_ds,
        batch_size,
        size_testset,
        type_partition,
        num_members,
        num_non_members,
        num_cover
    )
    
    print(f'attacker dataset size: {len(train_sets[0])}\nnormal user dataset size: {len(train_sets[1])}')
    print(f'cover set size: {len(cover_set)}')
    
    print(type(train_sets[0]), type(train_sets[1]))
    member_set, non_member_set = split_train_set(train_sets[0])
    attacker_train_set = shuffle_labels(train_sets[0])
    
    print(len(attacker_train_set), len(member_set), len(non_member_set))
    
    cur_cover_set = get_random_coverset(cover_set, 500)
    print(f'cur cover loader size: {len(cur_cover_set)}')
    
    print(f'x shape: {x_shape}')
    
    print(f'user data loader len: {len(train_sets[1])}')