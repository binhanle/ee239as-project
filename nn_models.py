import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions, state_shape, fc=512, kernel1=8, stride1=4, f1=32,
            kernel2=4, stride2=2, f2=64, kernel3=3, stride3=1, f3=64):
        super(DQN, self).__init__()

        c = state_shape[0]
        h = state_shape[1]
        w = state_shape[2]

        self.act = F.relu # relu activation function

        # 3 conv layers
        self.conv1 = nn.Conv2d(c, f1, kernel_size=kernel1, stride=stride1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel2, stride=stride2)
        self.bn2 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=kernel3, stride=stride3)
        self.bn3 = nn.BatchNorm2d(f3)

        # used for computing the output size of the conv layers
        def conv2d_size_out(size, kernels, strides):
            for k, s in zip(kernels, strides):
                size = (size - k) // s  + 1
            return size

        convw = conv2d_size_out(w, [kernel1, kernel2, kernel3], [stride1, stride2, stride3])
        convh = conv2d_size_out(h, [kernel1, kernel2, kernel3], [stride1, stride2, stride3])
        linear_input_size = convw * convh * f3

        # fully connected layers
        self.fc_layer = nn.Linear(linear_input_size, fc)
        self.out_layer = nn.Linear(fc, num_actions)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc_layer(x))
        x = self.out_layer(x)
        return x

class DQN_alternate(nn.Module):
    def __init__(self, num_actions, state_shape, fc=512, kernel1=8, stride1=4, f1=32,
            kernel2=4, stride2=2, f2=64, kernel3=3, stride3=1, f3=64, kernel4=3, stride4=1, f4=64):
        super(DQN, self).__init__()

        c = state_shape[0]
        h = state_shape[1]
        w = state_shape[2]

        self.act = F.relu # relu activation function

        # 3 conv layers
        self.conv1 = nn.Conv2d(c, f1, kernel_size=kernel1, stride=stride1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel2, stride=stride2)
        self.bn2 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=kernel3, stride=stride3)
        self.bn3 = nn.BatchNorm2d(f3)
        self.conv4 = nn.Conv2d(f3, f4, kernel_size=kernel4, stride=stride4)
        self.bn4 = nn.BatchNorm2d(f4)

        # used for computing the output size of the conv layers
        def conv2d_size_out(size, kernels, strides):
            for k, s in zip(kernels, strides):
                size = (size - k) // s  + 1
            return size

        convw = conv2d_size_out(w, [kernel1, kernel2, kernel3, kernel4], [stride1, stride2, stride3, stride4])
        convh = conv2d_size_out(h, [kernel1, kernel2, kernel3, kernel4], [stride1, stride2, stride3, stride4])
        linear_input_size = convw * convh * f4

        # fully connected layers
        self.fc_layer = nn.Linear(linear_input_size, fc)
        self.out_layer = nn.Linear(fc, num_actions)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc_layer(x))
        x = self.out_layer(x)
        return x