import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import importlib
import sys
from DL_attacks.utils import setup_data, setup_model
from DL_attacks.ops_on_vars_list import *

# 定义 ResBlock 和 ResNet20 类
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels) if bn else nn.Identity()
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet20(nn.Module):
    def __init__(self, input_shape, output_shape, bn=True):
        super(ResNet20, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) if bn else nn.Identity()
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(16, 16, 3, stride=1, bn=bn)
        self.layer2 = self._make_layer(16, 32, 3, stride=2, bn=bn)
        self.layer3 = self._make_layer(32, 64, 3, stride=2, bn=bn)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, output_shape)

    def _make_layer(self, in_channels, out_channels, blocks, stride, bn):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, bn))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1, bn=bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def binary_accuracy(label, p):
    predicted = torch.argmax(p, dim=1)
    correct_prediction = (predicted == label).float()
    return correct_prediction.mean()


if __name__ == "__main__":
    # # 数据预处理和加载
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    # test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    # test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    try:
        ds_setup_file = sys.argv[1]
        top_setup_file = sys.argv[2]
        run_num = sys.argv[3]
    except:
        print(f"USAGE: dataset_setup_file topology_setup_file run_number")
        sys.exit(1)
        
    
    Cds = importlib.import_module(ds_setup_file)
    Ctop = importlib.import_module(top_setup_file)

    # gets users' local training size
    size_local_ds = Cds.compute_local_training_set_size(Ctop.nu)
    
    print("Running setup ....")
    print(f'user dataset size: {size_local_ds}')
    # loads and splits local training sets and test one (validation)
    # NOTE: add cover set
    train_sets, test_set, cover_set, x_shape, num_class = setup_data(
        Cds.load_dataset,
        Ctop.nu,
        size_local_ds,
        Cds.batch_size,
        Cds.size_testset,
        Cds.type_partition
    )
    
    make_model = setup_model(
        Cds.model_maker,
        [x_shape, num_class, Ctop.init_lr, Ctop.lrd],
        Cds.model_same_init #! model_same_init: True --> 所有用户的模型初始化相同
    ) 
    
    # 模型定义
    input_shape = (3, 32, 32)
    output_shape = 10
    init_lr = 0.1
    step_slr = [25]
    # model, criterion, optimizer, scheduler, eval_metrics = resnet20(input_shape, output_shape, init_lr, step_slr)
    model1, criterion1, optimizer1, scheduler1, eval_metrics1 = make_model()
    optimizer1 = optimizer1(model1)
    scheduler1 = scheduler1(optimizer1)
    model2, criterion2, optimizer2, scheduler2, eval_metrics2 = make_model()
    optimizer2 = optimizer2(model2)
    scheduler2 = scheduler2(optimizer2)
    model1 = model1.cuda()
    model2 = model2.cuda()
    global_model = deepCopyModel(model1).cuda()
    # 训练和测试函数
    def train(epoch):
        model1.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_sets[1]):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer1.zero_grad()
            outputs = model1(inputs)
            loss = criterion1(outputs, targets)
            loss.backward()
            optimizer1.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:  # 每 100 个 iter 打印一次
                print(f'user 1 Epoch [{epoch + 1}], Step [{batch_idx + 1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        
        model2.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_sets[2]):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer2.zero_grad()
            outputs = model2(inputs)
            loss = criterion2(outputs, targets)
            loss.backward()
            optimizer2.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:  # 每 100 个 iter 打印一次
                print(f'user 2 Epoch [{epoch + 1}], Step [{batch_idx + 1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
        

    def test(model):
        
        # global_model_param = agg_sum_param(model1, model2)
        # global_model_param = agg_div_param(global_model_param, 2.0)
        
        # for p, op in zip(global_model.parameters(), global_model_param):
        #     p.data = op
            
        # global_model.eval()
        model.eval()
        correct = 0.0
        with torch.no_grad():
            for inputs, targets in test_set:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                correct += binary_accuracy(targets, outputs)
        print(f'Accuracy: {100 * correct / len(test_set):.2f}%')

    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        train(epoch)
        global_model_param = agg_sum_param(model1, model2)
        global_model_param = agg_div_param(global_model_param, 2.0)
        for p, op in zip(model1.parameters(), global_model_param):
            p.data = op
        for p, op in zip(model2.parameters(), global_model_param):
            p.data = op
        test(model1)
        test(model2)
        scheduler1.step()
        scheduler2.step()

    print("Finished Training")
