import math

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, einsum
from torch.autograd import Variable
from torch import linalg as LA

from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from einops import rearrange, repeat

from utils import set_parameter_requires_grad, get_vector_property

from model.modules1 import import_class, bn_init, EncodingBlock

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return scale



class FEM(nn.Module):
    def __init__(self, in_channels, channel_rate=2, reduction_ratio=16, num_action_classes=60):
        super(FEM, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // channel_rate
        self.inter_channels1 = in_channels
        if self.inter_channels == 0:
            self.inter_channels = 1

        # self.num_action_classes = num_action_classes

        self.common_v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)

        self.Trans_s = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.Trans_s[1].weight, 0)
        nn.init.constant_(self.Trans_s[1].bias, 0)

        self.Trans_q = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.Trans_q[1].weight, 0)
        nn.init.constant_(self.Trans_q[1].bias, 0)

        self.key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.dropout = nn.Dropout(0.1)
        self.ChannelGate = ChannelGate(self.in_channels, pool_types=['avg'], reduction_ratio=reduction_ratio)

        # self.action_embedding = nn.Embedding(num_action_classes, self.inter_channels)
        # self.action_attention = nn.Sequential(
        #     nn.Linear(1, self.inter_channels1),  # 1代表输入的浮点型数据
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.inter_channels1, self.inter_channels1)
        # )

    def forward(self, x):
        N,  C, T, V = x.shape

        # Common feature learning
        v_x = self.common_v(x).view(N,self.inter_channels, -1)
        v_x = v_x.permute(0, 2, 1)

        # action_attention = self.action_attention(action_class.unsqueeze(-1))
        # action_attention = torch.sigmoid(action_attention)

        # Attention calculation
        k_x = self.key(x).view(N, self.inter_channels, -1)
        k_x = k_x.permute(0, 2, 1)

        q_x = self.query(x).view(N, self.inter_channels, -1)

        A_x = torch.matmul(k_x, q_x)
        attention_x = F.softmax(A_x, dim=-1)

        # Feature aggregation and transformation
        p_x = torch.matmul(attention_x, v_x)
        p_x = p_x.permute(0, 2, 1).contiguous()
        p_x = p_x.view(N, self.inter_channels, T, V)
        p_x = self.Trans_s(p_x)
        p_x = self.ChannelGate(x) * p_x
        p_x = p_x + x


        # # Randomly select the first half of the batch dimension
        # random_indices = torch.randperm(N)[:N // 2]
        # p_x_selected = p_x[random_indices]

        # p_x = action_attention.unsqueeze(2).unsqueeze(3) * p_x

        return p_x

class ResidualConnection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ResidualConnection, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + residual)
        return x


#迭代注意特征融合模块
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.channel_residual = ResidualConnection(channels, channels )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xa = x + self.channel_residual(x)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + xa * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + xa * (1 - wei2)
        return xo


class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.channel_residual = ResidualConnection(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xa = x + self.channel_residual(x)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + xa * (1 - wei)
        return xi

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
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

class FAM(nn.Module):
    def __init__(self, in_c, out_c):
        super(FAM, self).__init__()
        # double conv
        self.conv0_1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(out_c)
        self.relu0_1 = nn.ReLU(inplace=True)

        self.conv0_2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(out_c)
        self.relu0_2 = nn.ReLU(inplace=True)

        # shortcut
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): # [2, 3, 256, 256]
        x0_1 = self.conv0_1(x)
        x0_1 = self.bn0_1(x0_1)
        x0_1 = self.relu0_1(x0_1) # [2, 64, 256, 256]

        x0_2 = self.conv0_2(x0_1)
        x0_2 = self.bn0_2(x0_2)
        x0_2 = self.relu0_2(x0_2) # [2, 64, 256, 256]

        x1 = self.conv1(x0_2)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.se(x2)

        x3 = self.conv3(x0_2)
        x3 = self.bn3(x3)
        # x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4



class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, num_head=3, noise_ratio=0.1, k=0, gain=1):
        super(InfoGCN, self).__init__()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)

        base_channel = 16
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        self.noise_ratio = noise_ratio
        self.z_prior = torch.empty(num_class, base_channel*4)
        self.A_vector = self.get_A(graph, k)
        self.gain = gain
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))

        self.l1 = EncodingBlock(base_channel, base_channel,A)
        self.l2 = EncodingBlock(base_channel, base_channel,A)
        self.l3 = EncodingBlock(base_channel, base_channel,A)

        self.l4 = EncodingBlock(base_channel, base_channel*2, A, stride=2)
        self.l5 = EncodingBlock(base_channel*2, base_channel*2, A)
        self.l6 = EncodingBlock(base_channel*2, base_channel*2, A)

        self.l7 = EncodingBlock(base_channel*2, base_channel*4, A, stride=2)
        self.l8 = EncodingBlock(base_channel*4, base_channel*4, A)
        self.l9 = EncodingBlock(base_channel*4, base_channel*4, A)

        # self.fem = FEM(base_channel * 4, reduction_ratio=4)
        # self.fam = FAM(base_channel*2, base_channel*2)
        self.aff1 = AFF(base_channel * 2, r=4)
        # self.aff2 = AFF(base_channel * 4, r=4)
        self.aff3 = iAFF(base_channel * 4, r=4)

        self.fc = nn.Linear(base_channel*4, base_channel*4)
        self.fc_mu = nn.Linear(base_channel*4, base_channel*4)
        self.fc_logvar = nn.Linear(base_channel*4, base_channel*4)
        self.decoder = nn.Linear(base_channel*4, num_class)
        nn.init.orthogonal_(self.z_prior, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.aff1(x)
        # x = self.fam(x)

        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)

        x = self.aff3(x)
        # x = self.fem(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        return y_hat, z
