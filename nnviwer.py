from torchsummary import summary
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384]).to("cuda")

summary(model, (3, 700, 700))  # 输出网络结构