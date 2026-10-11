import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from einops import rearrange
from torch import einsum


class MemoryUnit(nn.Module):
    def __init__(self, ptt_num, num_cls, part_num, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        """
        The instance PTT is divided into cls_number x ptt_number per cls x part number per ptt
        """
        self.num_cls = num_cls
        self.ptt_num = ptt_num
        self.part_num = part_num

        self.mem_dim = ptt_num * num_cls * part_num  # M
        self.fea_dim = fea_dim  # C
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres = shrink_thres

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.reweight_layer_part = nn.Conv1d(self.part_num, self.part_num, 1)
        self.reweight_layer_ins = nn.Conv1d(self.ptt_num, self.ptt_num, 1)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, residual=False):
        """
        This is a bottom-up hierarchical statistic and summarization module
        All steps in the main flow follow part -> prototype -> cls
        input = NHW x C
        total PTT M = num_cls (L) x ptt_num (T) x part_num (P)
        dimension C = fea_dim
        """

        # Part-unaware instance PTT (sub-flow)
        att_weight = F.linear(input, self.weight)  # NHW x M
        att_weight = F.softmax(att_weight, dim=1)  # NHW x M

        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output_part = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC

        # Part-aware instance PTT
        reweight_part = self.weight.view(self.num_cls * self.ptt_num, self.fea_dim, -1).permute(0, 2, 1)
        reweight_part = (torch.sigmoid(self.reweight_layer_part(reweight_part)) * reweight_part).permute(0, 2, 1)
        part_ins_att = self.avgpool(reweight_part).squeeze(-1)
        ins_att_weight = F.linear(output_part, part_ins_att)  # [NHW, C] x [C, M] = [NHW, M]
        ins_att_weight = F.softmax(ins_att_weight, dim=1)  # NHW x LT

        if self.shrink_thres > 0:
            ins_att_weight = hard_shrink_relu(ins_att_weight, lambd=self.shrink_thres)
            ins_att_weight = F.normalize(ins_att_weight, p=1, dim=1)

        ins_mem_trans = part_ins_att.permute(1, 0)  # Mem^T, MxC
        output_ins = F.linear(ins_att_weight, ins_mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC

        # Semantic PTT
        reweight_ins = part_ins_att.view(self.num_cls, self.ptt_num, self.fea_dim)
        reweight_ins = (torch.sigmoid(self.reweight_layer_ins(reweight_ins)) * reweight_ins).permute(0, 2, 1)
        sem_att = self.avgpool(reweight_ins).squeeze(-1)
        sem_att_weight = F.linear(output_ins, sem_att)
        sem_att_weight = F.softmax(sem_att_weight, dim=1)

        if self.shrink_thres > 0:
            sem_att_weight = hard_shrink_relu(sem_att_weight, lambd=self.shrink_thres)
            sem_att_weight = F.normalize(sem_att_weight, p=1, dim=1)

        sem_mem_trans = sem_att.permute(1, 0)
        output_sem = F.linear(sem_att_weight, sem_mem_trans)

        if residual:
            output_sem += output_ins  # Applying residual connection to the final output

        return {'output_sem': output_sem, 'output_part': output_part, 'output_ins': output_ins}


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class DocMemory(nn.Module):
    def __init__(self, ptt_num, num_cls, part_num, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(DocMemory, self).__init__()
        self.ptt_num = ptt_num
        self.num_cls = num_cls
        self.part_num = part_num
        ins_mem = False  # Flag to control whether to use part-level instance or global semantic memory
        if ins_mem:
            self.mem_dim = ptt_num * num_cls * part_num  # part-level instance
        else:
            self.mem_dim = num_cls  # global semantic
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.ptt_num, self.num_cls, self.part_num, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l != 4:
            print('Wrong feature map size')
            return None

        # Reshape the input tensor based on its dimensions
        x = input.permute(0, 2, 3, 1)

        x = x.contiguous()
        x = x.view(-1, s[1])

        # Pass the reshaped input through the memory unit
        y_and = self.memory(x)

        # Extract the outputs from the memory unit
        y_sem = y_and['output_sem']
        y_ins = y_and['output_ins']
        y_part = y_and['output_part']

        # Reshape the outputs back to their original dimensions
        y_sem = y_sem.view(s[0], s[2], s[3], s[1])
        y_sem = y_sem.permute(0, 3, 1, 2)
        y_ins = y_ins.view(s[0], s[2], s[3], s[1])
        y_ins = y_ins.permute(0, 3, 1, 2)
        y_part = y_part.view(s[0], s[2], s[3], s[1])
        y_part = y_part.permute(0, 3, 1, 2)

        return y_sem, y_ins, y_part


# ReLU based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
    return layer


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

##########################################################################
## Supervised Attention Module (RAM)
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img

##########################################################################
## Spatial Attention
class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y

# Spatial Attention Block (SAB)
class SAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(SAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)
        self.SA = SALayer(kernel_size=7)

    def forward(self, x):
        res = self.body(x)
        res = self.SA(res)
        res += x
        return res

##########################################################################
## Pixel Attention
class PALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias), # channel <-> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

## Pixel Attention Block (PAB)
class PAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(PAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.PA = PALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.PA(res)
        res += x
        return res

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
    
    
bn = 2  # block number-1


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    

class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

def to(x):
    return {'device': x.device, 'dtype': x.dtype}


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h



class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out
    

class SRTransformer(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads, spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type):
        super(SRTransformer, self).__init__()


        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x
    

class Encoder(nn.Module):
    def __init__(self, n_feat, bias, scale_unetfeats):
        super(Encoder, self).__init__()
        self.encoder_level1 = [SRTransformer(n_feat, 8, 0.5, 2, 2, 16, 2.66, bias, 'WithBias') for _ in range(bn)]
        self.encoder_level2 = [SRTransformer(n_feat + scale_unetfeats, 8, 0.5, 2, 2, 16, 2.66, bias, 'WithBias') for _ in range(bn)]
        self.encoder_level3 = [SRTransformer(n_feat + (scale_unetfeats * 2), 8, 0.5, 2, 2, 16, 2.66, bias, 'WithBias') for _ in range(bn)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)


    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, bias, scale_unetfeats):
        super(Decoder, self).__init__()
        self.decoder_level1 = [SRTransformer(n_feat, 8, 0.5, 2, 2, 16, 2.66, bias, 'WithBias') for _ in range(bn)]
        self.decoder_level2 = [SRTransformer(n_feat + scale_unetfeats, 8, 0.5, 2, 2, 16, 2.66, bias, 'WithBias') for _ in range(bn)]
        self.decoder_level3 = [SRTransformer(n_feat + (scale_unetfeats * 2), 8, 0.5, 2, 2, 16, 2.66, bias, 'WithBias') for _ in range(bn)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.skip_attn1 = SRTransformer(n_feat, 8, 0.5, 2, 2, 16, 2.66, bias, 'WithBias')
        self.skip_attn2 = SRTransformer(n_feat + scale_unetfeats, 8, 0.5, 2, 2, 16, 2.66, bias, 'WithBias')
        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1

##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
# Mixed Residual Module
class Mix(nn.Module):
    def __init__(self, m=1):
        super(Mix, self).__init__()
        w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2, feat3):
        factor = self.mix_block(self.w)
        other = (1 - factor)/2
        output = fea1 * other.expand_as(fea1) + fea2 * factor.expand_as(fea2) + feat3 * other.expand_as(feat3)
        return output
    

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
    

class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V
    

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, I):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.sk = SKFF(out_channels, height=2, reduction=8, bias=False)

    def forward(self, x):
        x1 = self.relu(self.conv(x))
        # output = torch.cat([x, x1], 1) # -> RDB
        output = self.sk((x, x1))
        return output
    

class SK_RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(SK_RDB, self).__init__()
        self.identity = nn.Conv2d(in_channels, growth_rate, 1, 1, 0)
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels, in_channels, I=i) for i in range(num_layers)]
        )
        self.lff = nn.Conv2d(in_channels, growth_rate, kernel_size=1)

    def forward(self, x):
        res = self.identity(x)
        x = self.layers(x)
        x = self.lff(x)
        return res + x
    

class StainRestorer(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=96, scale_unetfeats=48, kernel_size=3, bias=False):
        super(StainRestorer, self).__init__()

        self.memory = DocMemory(ptt_num=2, num_cls=10, part_num=5, fea_dim=in_c)

        # p_act = nn.PReLU()
        # self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
        #                                    conv(n_feat // 2, n_feat, kernel_size, bias=bias))
        # self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
        #                                    conv(n_feat // 2, n_feat, kernel_size, bias=bias))
        # self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
        #                                    conv(n_feat // 2, n_feat, kernel_size, bias=bias))

        self.patch_embed1 = OverlapPatchEmbed(in_c, n_feat)

        self.stage1_encoder = Encoder(n_feat, bias, scale_unetfeats)
        self.stage1_decoder = Decoder(n_feat, bias, scale_unetfeats)

        self.sam1o = SAM(n_feat, kernel_size=3, bias=bias)

        self.mix = Mix(1)

        self.tail = SK_RDB(in_channels=n_feat, growth_rate=out_c, num_layers=3)


    def forward(self, x):
        ## Compute Shallow Features
        shallow1, shallow2, shallow3 = self.memory(x)
        # shallow1 = self.shallow_feat1(x)
        # shallow2 = self.shallow_feat2(x)
        # shallow3 = self.shallow_feat3(x)

        mix_r = self.mix(shallow1, shallow2, shallow3)

        shallow1 = self.patch_embed1(mix_r)
        x1 = self.stage1_encoder(shallow1)
        x1_D = self.stage1_decoder(x1)
        ## Apply SAM
        x1_out, x1_img = self.sam1o(x1_D, x)
        x_final = self.tail(x1_out)

        return [x_final + x1_img, x1_img]
    

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = StainRestorer()
    from thop import profile
    macs, params = profile(model, inputs=(x,))
    print('macs: ', macs, 'params: ', params)
    print('macs: ', macs / 10 ** 9, 'G, params: ', params / 10 ** 6, 'M')