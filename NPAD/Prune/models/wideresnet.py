"""
Code adapted from https://github.com/xternalz/WideResNet-pytorch
Modifications = return activations for use in attention transfer,
as done before e.g in https://github.com/BayesWatch/pytorch-moonshine
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.relu = nn.ReLU()
        self.droprate = dropRate
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(out_planes)
            )

        # self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn1 = norm_layer(in_planes)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = norm_layer(out_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        # self.bn3 = norm_layer(out_planes)
        # self.droprate = dropRate
        # self.equalInOut = (in_planes == out_planes)
        # self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        #                        padding=0, bias=False) or None
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
        # if not self.equalInOut:
        #     x = self.relu1(self.bn1(x))
        # else:
        #     out = self.relu1(self.bn1(x))
        # out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        # if self.droprate > 0:
        #     out = F.dropout(out, p=self.droprate, training=self.training)
        # out = self.conv2(out)
        # return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, norm_layer=None):
        super(NetworkBlock, self).__init__()
        self._norm_layer = norm_layer
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor=1, dropRate=0.0, num_classes=10, norm_layer=None):
        super(WideResNet, self).__init__()
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        else:
            self._norm_layer = norm_layer
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = self._norm_layer(nChannels[0])
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, norm_layer=self._norm_layer)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, norm_layer=self._norm_layer)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, norm_layer=self._norm_layer)
        # global average pooling and classifier
        self.bn2 = self._norm_layer(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.block1(out)
        activation1 = out
        out = self.block2(out)
        activation2 = out
        out = self.block3(out)
        activation3 = out
        out = self.relu(self.bn2(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return activation1, activation2, activation3, self.fc(out)
        # return self.fc(out)
def select_model(dataset,
                 model_name='WRN-16-1',
                 pretrained=False,
                 pretrained_models_path=None,
                 n_classes=10):

    assert model_name in ['WRN-16-1', 'WRN-16-2', 'WRN-40-1', 'WRN-40-2', 'ResNet34', 'WRN-10-2', 'WRN-10-1']
    if model_name=='WRN-16-1':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='WRN-16-2':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name=='WRN-40-1':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='WRN-40-2':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == 'WRN-10-2':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == 'WRN-10-1':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='ResNet34':
        model = resnet(depth=32, num_classes=n_classes)
    else:
        raise NotImplementedError

    if pretrained:
        model_path = os.path.join(pretrained_models_path)
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        print(checkpoint.keys())
        model.load_state_dict(checkpoint)

        #print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})".format(model_path, checkpoint['epoch'], checkpoint['best_prec']))
        print("=> loaded checkpoint '{}' ".format(model_path))


    return model

def wideresnet16(num_classes=10, norm_layer=nn.BatchNorm2d):
    model = WideResNet(depth=16,  widen_factor=1, dropRate=0.0, num_classes=num_classes, norm_layer=norm_layer)
    return model
def wideresnet40(num_classes=10, norm_layer=nn.BatchNorm2d):
    model = WideResNet(depth=40,  widen_factor=2, dropRate=0.0, num_classes=num_classes, norm_layer=norm_layer)
    return model

if __name__ == '__main__':
    import random
    import time
    # from torchsummary import summary

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)
    model = wideresnet16()
    ### WideResNets
    # Notation: W-depth-wideningfactor
    # model = WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0.0)
    # model = WideResNet(depth=16, num_classes=10, widen_factor=2, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.0)
    #model = WideResNet(depth=16, num_classes=10, widen_factor=10, dropRate=0.0)
    #model = WideResNet(depth=22, num_classes=10, widen_factor=8, dropRate=0.0)
    #model = WideResNet(depth=34, num_classes=10, widen_factor=2, dropRate=0.0)
    #model = WideResNet(depth=40, num_classes=10, widen_factor=10, dropRate=0.0)
    # model = WideResNet(depth=40, num_classes=10, widen_factor=1, dropRate=0.0)
    # model = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.0)
    ###model = WideResNet(depth=50, num_classes=10, widen_factor=2, dropRate=0.0)


    t0 = time.time()
    output, _, __, ___ = model(x)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHPAE: ", output.shape)

    summary(model, input_size=(3, 32, 32))