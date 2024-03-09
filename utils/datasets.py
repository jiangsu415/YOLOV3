import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    # augment数据增强，normalized_labels 是否对标注框进行归一化。如果为 True，则标注框的坐标将被归一化到范围 [0, 1]
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100 # 图像中包含物体数量最多
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32 # 最小图像尺寸
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index): # 输入的是下标，输出的是图像路径，图像和标签

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip() # 在batch中分别读取每一条数据
        img_path = 'E:\\张紫扬01\\研01\\咕泡学习\\目标检测\\YOLO\\YOLOV3\\PyTorch-YOLOv3\\data\\coco' + img_path
        #print (img_path)
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB')) # 格式统一为RGB，再转化为tensor格式，因为PYtorch输入必须是tensor

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0) # 使用 unsqueeze(0) 将在第0维度（批次维度）上添加一个维度
            img = img.expand((3, img.shape[1:])) # 使用 expand 方法在图像的通道维度上进行扩展，使其具有3个通道。

        _, h, w = img.shape # 获取图像的长宽和通道维度
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1) # 是否图片已经经过标准化，长宽一致，如果self.normalized_labels为
                                                                         # True则为（h,w）,False则赋值为（1,1）
        # Pad to square resolution
        img, pad = pad_to_square(img, 0) # 把图片用0填充为正方形
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip() # 读取标签数据
        label_path = 'E:\\张紫扬01\\研01\\咕泡学习\\目标检测\\YOLO\\YOLOV3\\PyTorch-YOLOv3\\data\\coco\\labels' + label_path
        #print (label_path)

        targets = None
        if os.path.exists(label_path):
            # 类别，x,y,w,h
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5)) # 将txt格式的数据读进来，torch.from_numpy将array格式的数据转化为tensor格式
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)  # 第一个维度是检测目标的id 后面四个维度x,y,w,h为中心点的坐标
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)   # x-宽度的一般就是左上角x的坐标
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]  # 因为填充了0所以要把算出来的坐标加上
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w  # 重新获取中心点的坐标，是一个相对的值，所有取值都是相对值
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes   # 存到target中 [:, 1:]从第2列开始（即索引为1的列）到最后一列的所有元素进行切片

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)  # 做数据增强，图像镜面翻转

        return img_path, img, targets # 返回图像路径，图片数据和标签

    def collate_fn(self, batch): # 输入的是一个批次的图像数据（图像路径，图像和标签），输出的是对图片进行随机更改后的路径，图像和标签
        paths, imgs, targets = list(zip(*batch)) # 将批次中的数据解压成三个列表：paths 包含图像文件路径，imgs 包含图像数据，targets 包含标签信息。
        # 移除空的占位符目标。在之前的代码中，targets 被初始化为 None，表示一些图像可能没有标签信息。这里移除了那些没有标签信息的图像
        targets = [boxes for boxes in targets if boxes is not None]
        # 为每个目标框添加样本索引。在目标框的第一列（索引0）存储样本索引，表示该目标框属于哪个图像。
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0) # 将处理过的目标框信息连接成一个张量，以便后续的模型训练。
        # 如果启用了多尺度（multiscale），并且当前批次是每十批次中的第一批次，随机选择一个新的图像尺寸。这可以增加模型的鲁棒性和泛化能力。
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # 调整图像大小到指定的输入形状（self.img_size）。使用 resize 函数来实现。
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1 # 更新批次计数，以便在下一批次中选择是否改变图像尺寸。
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
