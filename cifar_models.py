"""CIFAR-10 adapted architectures for Assignment 2.

Compact, standard CIFAR versions (3x32x32 input, 10 classes) of the requested networks:
fully connected, LeNet-5, VGG19, ResNet18, SENet18, ResNeXt, DenseNet, GoogLeNet and DPN.
The designs follow the well-known CIFAR references (e.g. kuangliu/pytorch-cifar).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- Fully connected
class FCNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_classes))

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------- LeNet-5
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84), nn.ReLU(inplace=True), nn.Linear(84, num_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


# ----------------------------------------------------------------------------- VGG19
_VGG19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class VGG19(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        layers, c = [], 3
        for v in _VGG19:
            if v == 'M':
                layers += [nn.MaxPool2d(2)]
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


# ----------------------------------------------------------------------------- ResNet18 / SENet18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        self.se = se
        if se:
            self.fc1 = nn.Conv2d(planes, planes // 16 + 1, 1)
            self.fc2 = nn.Conv2d(planes // 16 + 1, planes, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se:
            w = F.adaptive_avg_pool2d(out, 1)
            w = F.relu(self.fc1(w)); w = torch.sigmoid(self.fc2(w))
            out = out * w
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, se=False):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, num_blocks[0], 1, se)
        self.layer2 = self._make(128, num_blocks[1], 2, se)
        self.layer3 = self._make(256, num_blocks[2], 2, se)
        self.layer4 = self._make(512, num_blocks[3], 2, se)
        self.linear = nn.Linear(512, num_classes)

    def _make(self, planes, n, stride, se):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s, se)); self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def ResNet18(num_classes=10):
    return ResNet([2, 2, 2, 2], num_classes, se=False)


def SENet18(num_classes=10):
    return ResNet([2, 2, 2, 2], num_classes, se=True)


# ----------------------------------------------------------------------------- ResNeXt
class ResNeXtBlock(nn.Module):
    def __init__(self, in_planes, cardinality, bottleneck_width, stride, expansion=2):
        super().__init__()
        width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, width, 1, bias=False); self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, expansion * width, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion * width)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != expansion * width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, expansion * width, 1, stride, bias=False),
                nn.BatchNorm2d(expansion * width))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    def __init__(self, num_blocks=(3, 3, 3), cardinality=8, bottleneck_width=16, num_classes=10):
        super().__init__()
        self.cardinality, self.bottleneck_width, self.expansion = cardinality, bottleneck_width, 2
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(num_blocks[0], 1)
        self.layer2 = self._make(num_blocks[1], 2)
        self.layer3 = self._make(num_blocks[2], 2)
        self.linear = nn.Linear(self.in_planes, num_classes)

    def _make(self, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(ResNeXtBlock(self.in_planes, self.cardinality, self.bottleneck_width, s))
            self.in_planes = self.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# ----------------------------------------------------------------------------- DenseNet (BC, CIFAR)
class DenseBottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([out, x], 1)


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)

    def forward(self, x):
        return F.avg_pool2d(self.conv(F.relu(self.bn(x))), 2)


class DenseNet(nn.Module):
    def __init__(self, nblocks=(6, 12, 24, 16), growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        gr = growth_rate
        planes = 2 * gr
        self.conv1 = nn.Conv2d(3, planes, 3, padding=1, bias=False)
        layers = []
        for i, nb in enumerate(nblocks):
            for _ in range(nb):
                layers.append(DenseBottleneck(planes, gr)); planes += gr
            if i != len(nblocks) - 1:
                out_planes = int(planes * reduction)
                layers.append(Transition(planes, out_planes)); planes = out_planes
        self.features = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(planes)
        self.linear = nn.Linear(planes, num_classes)

    def forward(self, x):
        out = self.features(self.conv1(x))
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1).flatten(1)
        return self.linear(out)


def densenet_cifar(num_classes=10):
    return DenseNet((6, 12, 24, 16), growth_rate=12, num_classes=num_classes)


# ----------------------------------------------------------------------------- GoogLeNet (CIFAR)
class Inception(nn.Module):
    def __init__(self, in_planes, n1, n3r, n3, n5r, n5, pool):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1, 1), nn.BatchNorm2d(n1), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3r, 1), nn.BatchNorm2d(n3r), nn.ReLU(True),
                                nn.Conv2d(n3r, n3, 3, padding=1), nn.BatchNorm2d(n3), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5r, 1), nn.BatchNorm2d(n5r), nn.ReLU(True),
                                nn.Conv2d(n5r, n5, 3, padding=1), nn.BatchNorm2d(n5), nn.ReLU(True),
                                nn.Conv2d(n5, n5, 3, padding=1), nn.BatchNorm2d(n5), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1),
                                nn.Conv2d(in_planes, pool, 1), nn.BatchNorm2d(pool), nn.ReLU(True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(True))
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.pre(x)
        out = self.b3(self.a3(out)); out = self.maxpool(out)
        out = self.b4(self.a4(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# ----------------------------------------------------------------------------- DPN (Dual Path, small)
class DPNBottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first):
        super().__init__()
        self.out_planes, self.dense_depth = out_planes, dense_depth
        self.conv1 = nn.Conv2d(last_planes, in_planes, 1, bias=False); self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes + dense_depth, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes + dense_depth)
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes + dense_depth, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes + dense_depth))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        s = self.shortcut(x); d = self.out_planes
        out = torch.cat([s[:, :d] + out[:, :d], s[:, d:], out[:, d:]], 1)
        return F.relu(out)


class DPN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        in_planes, out_planes = (96, 192, 384), (256, 512, 1024)
        num_blocks, dense_depth = (2, 2, 2), (16, 32, 24)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], 1)
        self.layer2 = self._make(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], 2)
        self.layer3 = self._make(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], 2)
        self.linear = nn.Linear(out_planes[2] + (num_blocks[2] + 1) * dense_depth[2], num_classes)

    def _make(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        layers = []
        for i, s in enumerate([stride] + [1] * (num_blocks - 1)):
            layers.append(DPNBottleneck(self.last_planes, in_planes, out_planes, dense_depth, s, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def build_models(num_classes=10):
    """Return the nine architectures as fresh instances."""
    return {
        "FC Net": FCNet(num_classes),
        "LeNet-5": LeNet5(num_classes),
        "VGG19": VGG19(num_classes),
        "ResNet18": ResNet18(num_classes),
        "SENet18": SENet18(num_classes),
        "ResNeXt": ResNeXt(num_classes=num_classes),
        "DenseNet": densenet_cifar(num_classes),
        "GoogLeNet": GoogLeNet(num_classes),
        "DPN": DPN(num_classes),
    }
