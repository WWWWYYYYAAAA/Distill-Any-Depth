
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import random
import os
# from mnist_dataset import MnistData
from MS_depth import MSNet, MSloss, MSNet_fix, DepthLoss
import time
import h5pickle
from torchvision.transforms import Compose
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from PIL import Image
from tqdm import tqdm
import random



class H5Dataset(Dataset):
    def __init__(self, file_path, image_key='images', label_key='depths'):
        self.file = h5pickle.File(file_path, 'r')
        self.images = self.file[image_key]
        self.labels = self.file[label_key]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.images[idx]).float()/255,
            torch.from_numpy(self.labels[idx]).float()/10
        )


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

outx = 160
outy = 120

resize_transform = transforms.Resize(
    size=(outx, outy),         # 目标尺寸
    interpolation=Image.BILINEAR  # 插值方法
)
resize_transform2 = transforms.Resize(
    size=(640, 480),         # 目标尺寸
    interpolation=Image.BILINEAR  # 插值方法
)
resize_IN = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 将数据集划分训练集和验证集
def split_data(files):
    """
    :param files:
    :return:
    """
    random.shuffle(files)
    # 计算比例系数，分割数据训练集和验证集
    ratio = 0.9
    offset = int(len(files) * ratio)
    train_data = files[:offset]
    val_data = files[offset:]
    return train_data, val_data


# 训练
def train(model, loss_func, optimizer, checkpoints, epoch):
    print('Train......................')
    # 记录每个epoch的loss和acc
    best_acc = 0
    best_loss = 100000
    best_epoch = 0
    # 训练过程
    for epoch in range(0, epoch):
        # 设置计时器，计算每个epoch的用时
        start_time = time.time()
        model.train()  # 保证每一个batch都能进入model.train()的模式
        # 记录每个epoch的loss和acc
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        # train_data =  train_data.to(device)
        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            # print(batch_size)
            # print(i, inputs, labels)
            rx = random.randint(0, 319)
            ry = random.randint(0, 239)
            Kr = random.randint(0, 1000)
            
            # Compose([
            #     # Resize(700, 700, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
            #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     PrepareForNet()
            # ])
            if Kr%3 <= 1 or random_mode:
                inputs = inputs[:,:,rx:rx+320,ry:ry+240]
                labels = labels[:,rx:rx+320,ry:ry+240]
                for i in range(inputs.shape[0]):
                    inputs[i,:,:,:] = resize_IN(inputs[i,:,:,:])
                inputs = resize_transform2(inputs)
                labels = resize_transform(labels)
            else:    
                for i in range(inputs.shape[0]):
                    inputs[i,:,:,:] = resize_IN(inputs[i,:,:,:])
                labels = resize_transform(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # transform = Compose([
            #     Resize(200, 200, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
            #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     PrepareForNet()
            # ])
            # Resize(80, 80)
            # 预测输出
            outputs = model(inputs).squeeze(dim=1)
            # print(outputs)
            # 计算损失
            # labels = labels.flatten(start_dim=1,end_dim=2)
            # print(outputs.shape, labels.shape)
            # loss = loss_func(outputs.view(outputs.shape[0], -1), labels.view(outputs.shape[0], -1))
            loss = loss_func(outputs, labels)
            # print(outputs)
            # 因为梯度是累加的，需要清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 优化器
            optimizer.step()
            # 计算准确率
            # output = nn.functional.softmax(outputs, dim=1)
            # pred = torch.argmax(output, dim=1)
            # acc = torch.sum(pred == labels)
            train_loss += loss.item()
            # train_acc += acc.item()
        # 验证集进行验证
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(val_data)):
                Kr = random.randint(0, 1000)
                if Kr%3<=1:
                    rx = random.randint(0, 319)
                    ry = random.randint(0, 239)
                    inputs = inputs[:,:,rx:rx+320,ry:ry+240]
                    labels = labels[:,rx:rx+320,ry:ry+240]
                    for i in range(inputs.shape[0]):
                        inputs[i,:,:,:] = resize_IN(inputs[i,:,:,:])
                    inputs = resize_transform2(inputs)
                    labels = resize_transform(labels)
                else:
                    for i in range(inputs.shape[0]):
                        inputs[i,:,:,:] = resize_IN(inputs[i,:,:,:])
                        labels = resize_transform(labels)

                inputs = inputs.to(device)
                labels = labels.to(device)
                # 预测输出
                outputs = model(inputs).squeeze(dim=1)
                # 计算损失
                loss = loss_func(outputs, labels)
                # 计算准确率
                output = nn.functional.softmax(outputs, dim=1)
                pred = torch.argmax(output, dim=1)
                # print(pred,'================')
                # print(pred==labels,'=====----------======')
                acc = torch.sum(pred == labels)
                # acc = calculat_acc(outputs, labels)
                val_loss += loss.item()
                val_acc += acc.item()
                if i >= (val_data_size-1):
                    break

        # 计算每个epoch的训练损失和精度
        train_loss_epoch = train_loss / train_data_size
        # train_acc_epoch = train_acc / train_data_size
        # 计算每个epoch的验证集损失和精度
        val_loss_epoch = val_loss / val_data_size
        # val_acc_epoch = val_acc / val_data_size
        end_time = time.time()
        print(
            'epoch:{} | time:{:.8f} | train_loss:{:.8f} | val_loss:{:.8f}'.format(
                epoch,
                end_time - start_time,
                train_loss_epoch,
                val_loss_epoch))
        # print(
        #     'epoch:{} | time:{:.8f} | train_loss:{:.8f} | train_acc:{:.8f}'.format(
        #         epoch,
        #         end_time - start_time,
        #         train_loss_epoch,
        #         train_acc_epoch))

        # 记录验证集上准确率最高的模型
        best_model_path = checkpoints + "/" + 'best_model' + '.pth'
        if val_loss_epoch <= best_loss:
            best_loss = val_loss_epoch
            best_epoch = epoch
            torch.save(model, best_model_path)
        torch.save(model, checkpoints + '/last.pth')
        print('Best loss for Validation :{:.8f} at epoch {:d}'.format(best_loss, best_epoch))
        # 每迭代50次保存一次模型
        # if epoch % 50 == 0:
        #     model_name = '/epoch_' + str(epoch) + '.pt'
        #     torch.save(model, checkpoints + model_name)
    # 保存最后的模型
    torch.save(model, checkpoints + '/last.pt')


if __name__ == '__main__':
    # batchsize
    # bs = 5000
    # learning rate
    lr = 0.00000001
    # epoch
    epoch = 10
    # checkpoints,模型保存路径
    checkpoints = 'MSNet'
    os.makedirs(checkpoints, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    random_mode = 0
    #load .mat
    file_path = "data/nyu_depth_v2_labeled.mat"
    dataset = H5Dataset(file_path)
    val_data_size = 64
    data_size = dataset.__len__()
    train_dataset = Subset(dataset, range(val_data_size,data_size))
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=6)
    
    val_dataset = Subset(dataset, range(val_data_size))
    val_data = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=6)
    train_data_size = train_dataset.__len__()
    # 加载模型
    model = torch.load(checkpoints+"/best_model.pth", weights_only=False)
    # model = MSNet_fix()
    
    # model.load_state_dict(torch.load('checkpoints/best_model.pth', weights_only=False))
    
    # GPU是否可用，如果可用，则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # outx = 160
    # outy = 120
    print(device)
    model.to(device)
    # 损失函数
    # loss_func = MSloss
    # loss_func = nn.CrossEntropyLoss()
    loss_func = DepthLoss
    # loss_func = nn.L1Loss()
    # 优化器，使用SGD,可换Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 训练
    train(model, loss_func, optimizer, checkpoints, epoch)