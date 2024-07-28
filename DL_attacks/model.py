import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
import torchvision

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

class LinearCIFAR10(nn.Module):
    def __init__(self):
        super(LinearCIFAR10, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 10)  # 3*32*32是输入的展平向量维度，10是CIFAR-10的类别数

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # 展平输入
        x = self.fc1(x)
        return x
    
def binary_accuracy(label, p):
    predicted = torch.argmax(p, dim=1)
    correct_prediction = (predicted == label).float()
    return correct_prediction.mean()

def resnet20(input_shape, output_shape, init_lr, step_slr: list, pretrain=False, checkpoint_path=None, bn=True):
    model = ResNet20(input_shape, output_shape, bn)
    if pretrain:
        if checkpoint_path is None:
            raise ValueError('pretrain need checkpoint path')
        else:
            ckpt = torch.load(checkpoint_path)
            for key, value in ckpt['net'].items():
                if 'module.' in key:
                    key = key.replace('module.', '')
            print(ckpt['net'].keys())
                    
            model.load_state_dict(ckpt['net'])
            print('load pre-trained checkpoint successfully')
    loss = nn.CrossEntropyLoss()
    optimizer_fn = lambda model: optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    scheduler_fn = lambda optimizer: MultiStepLR(optimizer, step_slr, gamma=0.1)
    
    return model, loss, optimizer_fn, scheduler_fn, binary_accuracy

def Linear(input_shape, output_shape, init_lr, step_slr: list, bn=True):
    model = ResNet20(input_shape, output_shape, bn)
    # model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    loss = nn.CrossEntropyLoss()
    optimizer_fn = lambda model: optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    scheduler_fn = lambda optimizer: MultiStepLR(optimizer, step_slr, gamma=0.1)
    
    return model, loss, optimizer_fn, scheduler_fn, binary_accuracy


if __name__ == "__main__":
    input_shape = (1, 3, 32, 32)
    output_shape = 10
    model, *out = resnet20(input_shape=input_shape, output_shape=output_shape, init_lr=0.1, step_slr=10)
    model = model.cuda()
    input = torch.rand(size=input_shape).cuda()
    output = model(input)
    print(output.shape)
    print(output.device)
    