from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs): # 输入的是读取后的配置文件，输出的是模型的超参数和模型列表
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)  # 获取到所有的超参数，
    output_filters = [int(hyperparams["channels"])]
    # 通常在构建神经网络时用于存储网络的各个层或模块。例如，如果有卷积层、全连接层、激活函数等，都可以添加到 module_list 中。这使得管理网络的参数和操作变得方便。
    module_list = nn.ModuleList()  # 他相当于一个list块，网络模型中的每一个结构都是一个list
    for module_i, module_def in enumerate(module_defs): # 遍历list
        modules = nn.Sequential()                   # 3合1，相当于是一个卷积层+BN+ReLu

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])  # 读取BN
            filters = int(module_def["filters"])     # 卷积的个数（特征图个数）
            kernel_size = int(module_def["size"])    # 卷积的大小
            pad = (kernel_size - 1) // 2             # padding填充
            modules.add_module(  # 卷积
                f"conv_{module_i}",  # 相当于卷积的api
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn: # BN # momentum 是用于计算滑动平均的动量参数，eps 是防止分母为零的小值
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":  # 激活函数
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":  # 没有maxpool
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1: # 这个条件检查是否需要在特殊情况下添加零填充。在某些情况下，为了保持特定形状的输出，需要手动添加零填充
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1))) # 如果需要添加零填充，这行代码将一个 nn.ZeroPad2d 层添加到模型中，实现指定的填充。
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":  # 只是定义上采样层，空的层
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest") # 创建一个上采样层，其中 scale_factor 指定上采样的倍数，而 mode 指定了上采样的方法，这里使用的是 "nearest"，即最近邻插值。
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route": # 输入1：26*26*256 输入2：26*26*128  输出：26*26*（256+128）进行一个拼接的操作
            layers = [int(x) for x in module_def["layers"].split(",")] # -4表示和前面第四层进行拼接
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())  # 只创建了一个空的层，只是定义

        elif module_def["type"] == "shortcut":  # 残差层，两条路径进行一个加法的操作
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer()) # 将一个名为 "shortcut_{module_i}" 的空层（EmptyLayer）添加到模型的模块列表中。这个空层实际上不执行任何计算，只是为了在模型结构中表示存在一个快捷连接

        elif module_def["type"] == "yolo":  # 一共有三个yolo层，三种不同的类别候选框计算损失值
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]  # 每个类别的候选框有三种先验框分别拿到他们的id
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")] # 拿到实际的数值
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)] # 把框做成hw的比例  # 将锚点列表重新组织为由每个锚点的宽度和高度组成的元组列表。
            anchors = [anchors[i] for i in anchor_idxs] # 最后得到yolo层实际的先验框 # 根据 mask 属性中的索引列表，从所有锚点中选择被选中的锚点。
            num_classes = int(module_def["classes"]) # 输入的种类
            img_size = int(hyperparams["height"])   # 输入图像的大小
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size) #  创建一个YOLOLayer，该层使用选定的锚点、类别数和图像大小初始化
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)  # 第一个模块添加进来
        output_filters.append(filters)  # 把最终输出的特征图个数添加进来

    print(module_list)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors  # 先验框的大小
        self.num_anchors = len(anchors)  # 先验框的数量
        self.num_classes = num_classes   # 类别
        self.ignore_thres = 0.5          # 阈值
        self.mse_loss = nn.MSELoss()     # 损失函数
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size # 网格个数
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size  # 获得下采样的次数倍数32
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor) # 分别赋值对这15行0,1,2.。。。
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]) # 将先验框的大小也缩小为32倍，当前格子中是多大的
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1)) # 当前候选框的w,h分别计算出来
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):  # 输入的是预测的图片,输出是输出结果和总体的损失
        # Tensors for cuda support
        print (x.shape)  # [4,255,15,15]，4指的是batch是4,255，得到的值是255特征图个数，15*15特征图大小
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim  # 当前输入图像大小，训练时会随机一个图像大小，为了使网络能够适应不同的分辨率的，除的开32
        num_samples = x.size(0)  # 输入图像的batch=4,一次训练4张图像
        grid_size = x.size(2)    # 当前网格的大小，是由输入图像大小决定的480/32

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size) # 进行一个维度的变化
            # num_samples为batch，self.num_anchors先验框的个数，每个点对应三个先验框，self.num_classes + 5：80个类别加x,y,w,h还有置信度
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        print (prediction.shape)
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x 每个框中心点的坐标
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf  置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. 通过sigmoid函数预测80个类别，每一个类别属于当前类别的可能性是多少

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda) # 相对位置得到对应的绝对位置比如之前的位置是0.5,0.5变为 11.5，11.5这样的

        # Add offset and scale with anchors #特征图中的实际位置
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x # 将预测框的中心点 x 坐标加上相对位置的偏移，使用 grid_x 表示网格的绝对位置。
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w # 得到实际的x,y,w,h的值
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride, # 还原到原始图中，将box还原到原来大小，输出最终的张量
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )
            # iou_scores：真实值与最匹配的anchor的IOU得分值 class_mask：分类正确的索引
            # obj_mask：目标框所在位置的最好anchor置为1 noobj_mask obj_mask那里置0，还有计算的iou大于阈值的也置0，其他都为1
            # tx, ty, tw, th, 对应的对于该大小的特征图的xywh目标值也就是我们需要拟合的值 tconf 目标置信度
            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask]) # 只计算有目标的 # 差的平方
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask]) # LOSS=−(ylog(p(x)+(1−y)log(1−p(x)) p(x)模型输出，y是真实值
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj # 有物体越接近1越好 没物体的越接近0越好
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask]) #分类损失
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls #总损失

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path) # 读取配置文件
        self.hyperparams, self.module_list = create_modules(self.module_defs) # 按照配置文件顺序逐层配置好结构
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):  # 输入的是图像，输出是yolo层的输出结果或者（yolo层输出结果和损失）
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []  # 当前层的输出结果，以及yolo层输出的结果
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)   # 经过一次卷积后的结果
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1) # 拼接
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]  # 将当前层的结果（-1）和前面第三层（-3）的结果做一个拼接
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim) # x是前一层的结果，targets标签，img_dim输入图像的大小
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x) # 例如卷积后一层结果放到这里面
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
