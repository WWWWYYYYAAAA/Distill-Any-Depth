import torch
import torch.nn as nn
from torchsummary import summary

class MSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #coarse
        self.coarse1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.coarse2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.coarse345 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.fine1 = nn.Sequential(
            nn.Conv2d(3, 63, kernel_size=9, stride=2, padding=0),
            nn.BatchNorm2d(63),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.flat = nn.Flatten()
        self.fine3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fine4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
    
    def forward(self, x):
        xc = self.coarse1(x)
        xc = self.coarse2(xc)
        xc = self.coarse345(xc)
        xc = self.flat(xc)
        self.fc1 = nn.Linear(xc.shape[1], 4096).to(xc.device)
        # print(xc.device)
        # print(xc.shape)
        xc = self.fc1(xc)
        xf = self.fine1(x)
        self.fc2 = nn.Linear(4096, xf.shape[2]*xf.shape[3]).to(xc.device)
        # print(xf.shape)
        
        xc = self.fc2(xc)
        xc = xc.view(-1,1, xf.shape[2], xf.shape[3])
        xf = torch.cat([xf, xc], dim = 1)
        # print(xf.shape, xf.size(0))
        xf = self.fine3(xf)
        xf = self.fine4(xf)
        return xf
    
class MSNet_fix(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #coarse
        self.coarse1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.coarse2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.coarse345 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.fine1 = nn.Sequential(
            nn.Conv2d(3, 63, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(63),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.flat = nn.Flatten()
        self.fine3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fine4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.fc1 = nn.LazyLinear(4096)
        self.fc2 = nn.Linear(4096, 160*120)
    
    def forward(self, x):
        xc = self.coarse1(x)
        xc = self.coarse2(xc)
        xc = self.coarse345(xc)
        xc = self.flat(xc)
        # print(xc.device)
        # print(xc.shape)
        xc = self.fc1(xc)
        xf = self.fine1(x)
        # print(xf.shape)
        
        xc = self.fc2(xc)
        xc = xc.view(-1,1, 160, 120)
        xf = torch.cat([xf, xc], dim = 1)
        # print(xf.shape)
        # print(xf.shape, xf.size(0))
        xf = self.fine3(xf)
        xf = self.fine4(xf)
        # print(xf.shape)
        return xf
    
def MSloss(predictions, targets, lmd = 0.5,device='cuda'):
    pf = predictions.view(-1)
    # print("pf ", pf)
    tf = targets.view(-1)
    # print("tf ", tf.max())
    # di = torch.log((pf+1e6)*255) - torch.log((tf+1e6)*255)
    di = torch.abs(pf-tf)
    # print(di)
    lyy = torch.mean(di**2) - lmd*(torch.mean(di)**2)
    return lyy
    
def DepthLoss(pred, target, alpha=0.5):
    # 1. L1损失（基础精度）
    lossf =  nn.L1Loss()
    l1_loss = lossf(pred, target)
    
    # 2. 梯度差异损失（提升边缘锐度）
    grad_x_pred = pred[:, :, :-1] - pred[:, :, 1:]
    grad_y_pred = pred[:, :-1, :] - pred[:, 1:, :]
    grad_x_target = target[:, :, :-1] - target[:, :, 1:]
    grad_y_target = target[:, :-1, :] - target[:, 1:, :]
    
    grad_loss = lossf(grad_x_pred, grad_x_target) + \
                lossf(grad_y_pred, grad_y_target)
    
    # 3. 组合损失
    return (1 - alpha) * l1_loss + alpha * grad_loss
    
if __name__ == '__main__':
# 测试数据
    x = torch.rand((2, 3, 304, 228))
    cnn = MSNet_fix()
    # print(cnn)
    # out = cnn(x)
    # print(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = cnn.to(device)
    summary(cnn, (3, 640, 480))
    
    loss = MSloss(torch.rand((2, 3, 304, 228)), torch.rand((2, 3, 304, 228)))
    print(loss)
    lossfunc = nn.SmoothL1Loss(reduction='mean')
    lossfunc1 = nn.MSELoss()
    loss2 = lossfunc1(torch.rand((2, 3, 304, 228)), torch.rand((2, 3, 304, 228)))
    print(loss2)