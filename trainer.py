import dataset
# from darknet53 import MainNet
from mobilenet_darknet53 import MainNet
import torch
import torch.nn as nn
import os
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.tensorboard import SummaryWriter

def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)            #换轴
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)      #重新reshape成标签格式
    label = target.float()                  #得到标签
    mask_obj = target[..., 4] > 0           #最后一维0索引大于0的掩码
    mask_noobj = target[..., 4] == 0        #最后一维0索引等于0的掩码

    output_ , target_ = output[mask_obj],label[mask_obj]             #得到正样本的输出和标签
    output_noobj , target_noobj = output[mask_noobj] , label[mask_noobj]             #得到负样本的输出和标签

    target1 = target_[:,5:85]              #分类标签
    _,pred = torch.max(target1,dim=1)         #变成标量，因为onehot不能传入CrossEntropyLoss()
    loss_cls = cls_loss(torch.sigmoid(output_[:, 4]),target_[:, 4])         #正样本置信度损失
    loss_xy = cls_loss(torch.sigmoid(output_[:, 0:2]),target_[: , 0:2])      #正样本中心点损失
    loss_cls_off = torch.mean((output_[:,2:4] - target_[:,2:4]) ** 2)     #正样本回归框W、H损失
    loss_class = corr_loss(output_[:,5:85],pred)          #正样本分类损失

    target2 = target_noobj[:, 5:85]                 #分类标签
    _, pred_noobj = torch.max(target2, dim=1)       #变成标量

    loss_cls_ = cls_loss(torch.sigmoid(output_noobj[:, 4]),target_noobj[:, 4])            #负样本置信度损失
    loss_xy_ = cls_loss(torch.sigmoid(output_noobj[:, 0:2]), target_noobj[:, 0:2])        #负样本中心点损失
    loss_cls_off_ = torch.mean((output_noobj[:, 2:4] - target_noobj[:, 2:4]) ** 2)        #负样本回归框W、H损失
    loss_class_ = corr_loss(output_noobj[:,5:85], pred_noobj)            #负样本分类损失

    loss_obj = loss_cls_off + loss_class + loss_cls + loss_xy          #正样本总损失
    loss_noobj = loss_cls_off_ + loss_class_ + loss_cls_ + loss_xy_     #负样本总损失

    loss = alpha *loss_obj + (1 - alpha) * loss_noobj       #加到一起优化，负样本比例比较高，加个alpha系数均衡正负样本

    return loss        #返回总损失

save_path = r"model\YOLO_net.pth"       #模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corr_loss = nn.CrossEntropyLoss()
cls_loss = nn.BCELoss()

# writer = SummaryWriter()
if __name__ == '__main__':

    myDataset = dataset.MyDataset()        #加载数据集
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=10, shuffle=True)

    net = MainNet()   #实例化网络
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))       #加载网络模型

    net.train().to(device)

    opt = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0005)     #优化器
    scheduler = lr_scheduler.StepLR(opt, 10, gamma=0.8)              #学习率调整

    loss_ = 10
    ecoph = 0
    while True:
        scheduler.step()
        for i,(target_13, target_26, target_52, img_data) in enumerate(train_loader):

            target_13 = target_13.to(device)
            target_26 = target_26.to(device)
            target_52 = target_52.to(device)
            img_data = img_data.to(device)

            output_13, output_26, output_52 = net(img_data)               #前向计算
            loss_13 = loss_fn(output_13, target_13, 0.95)
            loss_26 = loss_fn(output_26, target_26, 0.95)
            loss_52 = loss_fn(output_52, target_52, 0.95)

            loss = loss_13 + loss_26 + loss_52
            # loss_.append(loss.item())
            opt.zero_grad()         #损失优化
            loss.backward()
            opt.step()
            # if i % 10 == 0:
            #     writer.add_scalar('loss', loss, len(train_loader) * ecoph + i)
            print("ecoph:{}  {}/{}  loss:{}  save_loss:{}".format(ecoph,i,len(train_loader),loss.item(),loss_))
            if loss.item() <= loss_:
                torch.save(net.state_dict(),save_path)
                loss_ = loss.item()
                print("模型保存成功！")
        ecoph += 1
