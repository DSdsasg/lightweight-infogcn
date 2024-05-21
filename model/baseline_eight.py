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

from model.modules1 import import_class, bn_init, EncodingBlock, SA_GC


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

class AFFWithCustomGCN(nn.Module):
    def __init__(self, channels=32, r=4, gcn_channels=32, num_head=3, A=None):
        super(AFFWithCustomGCN, self).__init__()
        self.aff = AFF(channels, r)
        self.custom_gcn = SA_GC(channels, gcn_channels, A)  # 使用自定义的 GCN

    def forward(self, x, attn=None):
        # 输入 x 是节点特征
        x = self.aff(x)  # 应用 AFF
        gcn_output = self.custom_gcn(x, attn)  # 应用自定义 GCN
        return F.relu(gcn_output)  # 结合 AFF 和 GCN 输出

class AFFWithCustomGCN1(nn.Module):
    def __init__(self, channels=32, r=4, gcn_channels=32, num_head=3, A=None):
        super(AFFWithCustomGCN, self).__init__()
        self.aff = AFF(channels, r)
        self.custom_gcn = SA_GC(channels, gcn_channels, A)  # 使用自定义的 GCN

    def forward(self, x, attn=None):
        # 输入 x 是节点特征
        x = self.aff(x)  # 应用 AFF
        gcn_output = self.custom_gcn(x, attn)  # 应用自定义 GCN
        return F.relu(gcn_output)

class iAFFWithCustomGCN(nn.Module):
    def __init__(self, channels=32, r=4, gcn_channels=64, num_head=3, A=None):
        super(iAFFWithCustomGCN, self).__init__()
        self.iaff = iAFF(channels, r)
        self.custom_gcn = SA_GC(channels, gcn_channels, A)  # 使用自定义的 GCN
        self.se = SELayer(channels)
    def forward(self, x, attn=None):
        # 输入 x 是节点特征
        x = self.custom_gcn(x, attn)
        x = self.se(x)
        x = self.iaff(x)  # 应用 AFF

          # 应用自定义 GCN
        return F.relu(x)  # 结合 AFF 和 GCN 输出


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
        # self.l2 = EncodingBlock(base_channel, base_channel,A)
        # self.l3 = EncodingBlock(base_channel, base_channel,A)

        self.l4 = EncodingBlock(base_channel, base_channel*2, A, stride=2)
        # self.l5 = EncodingBlock(base_channel*2, base_channel*2, A)
        # self.l6 = EncodingBlock(base_channel*2, base_channel*2, A)


        self.l7 = EncodingBlock(base_channel*2, base_channel*4, A, stride=2)
        self.l8 = EncodingBlock(base_channel*4, base_channel*4, A)
        self.l9 = EncodingBlock(base_channel*4, base_channel*4, A)


        self.aff = AFFWithCustomGCN(base_channel , r=2, gcn_channels=16, num_head=3, A=A)
        self.aff1 = AFFWithCustomGCN(base_channel * 2, r=4, gcn_channels=32, num_head=3, A=A)
        self.aff2 = iAFFWithCustomGCN(base_channel * 4, r=4, gcn_channels=64, num_head=3, A=A)
        self.aff3 = AFFWithCustomGCN(base_channel * 4, r=4, gcn_channels=64, num_head=3, A=A)
        self.se = SELayer(base_channel)
        self.se1 = SELayer(base_channel * 2)
        self.se2 = SELayer(base_channel * 4)
        # self.aff3 = iAFF(base_channel * 4, r=4)

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
        x = self.aff(x)
        x = self.se(x)

        x = self.l4(x)
   
        x = self.aff1(x)
        x = self.se1(x)


        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)

        x = self.aff3(x)
        x = self.se2(x)


        x = self.aff2(x)




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
