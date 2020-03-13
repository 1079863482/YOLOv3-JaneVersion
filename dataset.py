from torch.utils.data import Dataset,DataLoader
import torchvision
import cfg
import os
from utils import *
from PIL import Image
import math

LABEL_FILE_PATH = r"X:\celeba\label_50000.txt"     #标签txtlujing
IMG_BASE_DIR = "X:\celeba\celeba_50000"      #图片路径

transforms = torchvision.transforms.Compose([         #归一化，Tensor处理
    torchvision.transforms.ToTensor()
])


def one_hot(cls_num, v):             #将分类数目转化为onehot编码
    b = np.zeros(cls_num)
    b[v] = 1.
    return b


class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()     #逐行加载

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}
        line = self.dataset[index]       #按照索引取出标签
        strs = line.split()     #分割
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))     #取出图片文件名并加入数据集中
        img_data_ = _img_data.copy()
        W,H,scale,img = narrow_image(_img_data)
        img_data = transforms(img)
        _boxes = np.array([float(x) for x in strs[1:]])        #将文件名后的cls, cx, cy, w, h 转化为浮点数
        boxes = np.split(_boxes, len(_boxes) // 5)             #标签每5个为一组
        boxes = np.stack(boxes)
        boxes = narrow_box(W,H,scale,boxes)
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():        #取出对应特征图的建议框
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))    #生成标签格式的4维0矩阵，例：（13，13，3，15），然后将正样本插入第4维度

            for box in boxes:     #取出每组标签实际框
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)     #计算中心点X坐标在原图对应的方格位置，取小数位，整数位为13x13方格索引
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)     #计算中心点Y坐标在原图对应的方格位置，取小数位，整数位为13x13方格索引

                for i, anchor in enumerate(anchors):       #依次取出三个建议框
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]    #根据索引取出三个建议框的面积
                    p_w, p_h = w / anchor[0], h / anchor[1]             #w和h分别除以建议框的w，h得到缩放比例
                    p_area = w * h                                      #实际框的面积
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)         #计算实际框和建议框的面积iou比
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(            #将得到的偏移量插入上面的四维矩阵中，得到成样本，其余的为负样本
                        [cx_offset, cy_offset, np.log(p_w), np.log(p_h), iou,* one_hot(cfg.CLASS_NUM, int(cls))])

        return labels[13], labels[26], labels[52], img_data


if __name__ == "__main__":
    data = MyDataset()
    train = DataLoader(data,1,True)
    for i , (a,b,c,d) in enumerate(train):
        print(a.size())
        print(b.size())
        print(c.size())
        print(d.size())
        # break
    