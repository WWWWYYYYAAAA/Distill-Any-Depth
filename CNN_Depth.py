import torch
import torch.nn as nn
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5,
                               padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 处理维度不匹配的跳跃连接

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)  # 调整输入维度
        out += identity  # 残差连接
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, outsize=80):
        super().__init__()
        self.in_channels = 64
        self.outsize = outsize
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 残差块堆叠
        self.layer1 = self._make_layer(self.in_channels, 2, stride=1)
        # self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(self.in_channels, 2, stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((20, 20))  # 替代原全连接层
        self.fc = nn.Linear(self.in_channels*20*20, self.outsize*outsize)
        self.avgpool2 = nn.AdaptiveAvgPool2d((self.outsize, self.outsize)) 
        self.finalrelu = nn.ReLU()
        # self.fc = nn.Linear(10000, 10000)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:  # 处理维度不匹配
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, 
                                   stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.finalrelu(x)
        # x = x.view(100, 100)
        # x = self.avgpool2(x)
        return x
    
if __name__ == '__main__':
# 测试数据
    x = torch.rand((2, 3, 28, 28))
    cnn = ResNet(outsize=120)
    print(cnn)
    out = cnn(x)
    print(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = cnn.to(device)
    # 网络模型的数据流程及参数信息
    summary(cnn, (3, 80, 80))
