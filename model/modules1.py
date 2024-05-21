import math
import os

import torch
import numpy as np

from torch import nn, einsum
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from einops import rearrange, repeat

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(UnitTCN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.se(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_heads):
        super(SelfAttention, self).__init__()
        self.scale = hidden_dim ** -0.5
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim*2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)


    def forward(self, x):
        y = rearrange(x, 'n c t v -> n t v c').contiguous()
        y = self.ln(y)
        y = self.to_qk(y)
        qk = y.chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b t v (h d) -> (b t) h v d', h=self.n_heads), qk)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k)*self.scale
        attn = dots.softmax(dim=-1).float()


        return attn

class SA_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(SA_GC, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head= A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        rel_channels = in_channels // 8
        # self.se = SELayer(in_channels)
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)


    def forward(self, x, attn=None):
        N, C, T, V = x.size()

        out = None
        if attn is None:
            # x = self.se(x)
            attn = self.attn(x)
        A = attn * self.shared_topology.unsqueeze(0)
        for h in range(self.num_head):
            A_h = A[:, h, :, :] # (nt)vv
            feature = rearrange(x, 'n c t v -> (n t) v c')
            z = A_h@feature
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        if channels < 32:
            factor = 8
        elif channels < 64:
            factor = 16
        else:
            factor = 32

        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # n*m, C , T ,V
        b, c, t, j = x.size()
        group_x = x.reshape(b * self.groups, -1, t, j)  # b*g,c//g,h,w
        x_t = self.pool_h(group_x)
        x_j = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_t, x_j], dim=2))
        x_t, x_j = torch.split(hw, [t, j], dim=2)
        x1 = self.gn(group_x * x_t.sigmoid() * x_j.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, tj
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, tj
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, t, j)

        # weights1 = self.sigmoid(weights)
        #
        # attention_map = weights1.detach().cpu().numpy()  # 转换为 NumPy 数组
        #
        # label = label.cpu().detach().numpy()
        # # 选择一个维度作为热图的输入（例如，选择第一个维度）
        # selected_map = attention_map.mean(axis=(2))
        #
        #
        # selected_map = selected_map[0]  # 选择第一个维度
        #
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(selected_map, annot=False, cmap='OrRd', cbar=True)
        # plt.title('Attention Map')
        # plt.xlabel('Attention Strength')
        # plt.ylabel('Joint Index')
        #
        # # 创建文件夹以保存热图
        # save_dir = 'attention_maps3'
        # os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(f'{save_dir}/attention_map_{label}.png')  # 保存热图到文件夹
        # plt.close()  # 关闭绘图

        return (group_x * weights.sigmoid()).reshape(b, c, t, j)

class SkeletonEMA(nn.Module):
    def __init__(self, in_channels, factor=32):
        super(SkeletonEMA, self).__init__()
        if in_channels < 32:
            factor = 8
        elif in_channels < 64:
            factor = 16
        elif in_channels < 128:
            factor = 32
        elif in_channels < 256:
            factor = 64
        else:
            factor = 128
        self.groups = factor
        assert in_channels // self.groups > 0
        self.softmax_t = nn.Softmax(-1)
        self.softmax_j = nn.Softmax(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(in_channels // self.groups, in_channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels // self.groups, in_channels // self.groups, kernel_size=3, stride=1, padding=1)
        # self.iaff = iAFF(in_channels, r=4)

    def forward(self, x):
        b, c, t, j = x.size()
        group_x = x.reshape(b * self.groups, -1, t, j)  # b*g, c//g, t, j

        # 在时间维度上计算注意力权重
        x_t = self.pool_h(group_x)
        hw_t = self.conv1x1(x_t)
        x_t = self.softmax_t(hw_t.reshape(b * self.groups, -1, t, 1))
        x_t = x_t.reshape(b * self.groups, -1, t, 1)

        # 在关节维度上计算注意力权重
        x_j = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw_j = self.conv1x1(x_j)
        x_j = self.softmax_j(hw_j.reshape(b * self.groups, -1, 1, j))
        x_j = x_j.reshape(b * self.groups, -1, 1, j)

        # 结合时间和关节维度的注意力权重
        weights = x_t * x_j
        weights = self.conv3x3(weights)
        weights = weights.reshape(b, c, t, j)
        # 对输入特征进行缩放
        x = x * weights.sigmoid()

        x = x.reshape(b, c, t, j)

        return x


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(EncodingBlock, self).__init__()
        self.agcn = SA_GC(in_channels, out_channels, A)
        self.tcn = MS_TCN(out_channels, out_channels, kernel_size=5, stride=stride,
                         dilations=[1, 2], residual=False)
        self.ema = EMA(out_channels)
        # self.se = SELayer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, attn=None):
        y = self.agcn(x, attn)
        y = self.tcn(y)
        y = self.ema(y)
        # y = self.se(y)
        y = self.relu(y + self.residual(x))

        return y

