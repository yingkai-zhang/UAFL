import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import DropPath, to_2tuple
from .DCN import DCN_Refine_With_Prior_flow

def mix(A_hat, E_y):
    bs, r, h, w = A_hat.shape
    _, _, c = E_y.shape
    
    # Reshape A_hat and E_y for batch matrix multiplication
    A_hat_f_m = A_hat.reshape(bs, r, h * w)  # Shape: (bs, r, h*w)
    E_y_reshaped = E_y.permute(0, 2, 1)  # Shape: (bs, c, r)
    
    # Perform batch matrix multiplication
    X_hat_m = torch.bmm(E_y_reshaped, A_hat_f_m)  # Shape: (bs, c, h*w)
    
    # Reshape the result back to (bs, c, h, w)
    X_hat = X_hat_m.reshape(bs, c, h, w)
    
    return X_hat

def Unmix_svd_3d(y, Rr = 3):
    
    # Reshape input tensor to be of shape (b, c, h*w)
    b, c, h, w = y.shape
    y = y.reshape(b, c, -1)
    
    # Perform SVD
    U, S, V = torch.svd(y)
    E = U[:, :, :Rr].permute(0, 2, 1)
    A = E @ y
    # print(E.shape)
    
    # Reshape A back to original shape
    A = A.reshape(b, Rr, h, w)
    
    return A, E



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):   
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

def shift_back(inputs,step=2):          # The shifting step varies with different HSI systems
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) 
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        output:(num_windows*B, N, C)
                """
        B_, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class CrossWindowAttention(nn.Module):
    r""" 基于窗口的多头交叉注意力（W-MCA）模块，带有相对位置偏置。支持移位和非移位窗口。
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义一个相对位置偏置的参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # 获取窗口内每个 token 的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        # --- 修改点: 将 qkv 分离为 q 和 kv ---
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # --- 修改结束 ---

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, ref, mask=None):
        """
        Args:
            x: 输入特征，形状为 (num_windows*B, N, C)
            ref: 参考特征，形状为 (num_windows*B, N, C)
            mask: (0/-inf) 的掩码, 形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None

        Returns:
            输出特征，形状为 (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        
        # --- 修改点: 分别从 x 和 ref 计算 q, k, v ---
        # 原始qkv计算: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # --- 修改点: Q, K, V 全部从 x 生成 ---
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # --- 修改结束 ---

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # --- 新增核心逻辑: 将 v 和 ref 逐元素相乘 ---
        # v 的形状:    (B_, num_heads, N, C // num_heads)
        # ref 的形状:  (B_, N, C)
        # 需要将 ref 变形以匹配 v
        ref_reshaped = ref.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        v_modulated = v * ref_reshaped # 逐元素相乘

        x = (attn @ v_modulated).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossSwinTransformerBlock(nn.Module):
    r""" Swin Transformer 交叉注意力模块.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution if isinstance(input_resolution, tuple) else (input_resolution, input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        
        # --- 修改点: 实例化 CrossWindowAttention ---
        self.attn = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # --- 修改结束 ---

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 为了代码完整性，这里保留MLP和norm2层，但在原始代码片段中它们被移除了
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(...) # 假设Mlp类已定义

        if self.shift_size > 0:
            # 计算注意力掩码
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            # 确保 window_partition 函数已定义
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, ref):
        """
        Args:
            x: 输入特征，形状为 (B, H, W, C)
            ref: 参考特征，形状为 (B, H, W, C)

        Returns:
            输出特征，形状为 (B, H, W, C)
        """
        B, H, W, C = x.shape
        x = self.norm1(x.view(B, H * W, C)).view(B, H, W, C)
        
        # --- 修改点: 对 ref 执行相同的操作 ---
        ref = self.norm1(ref.view(B, H * W, C)).view(B, H, W, C)
        # --- 修改结束 ---

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_ref = torch.roll(ref, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_ref = ref

        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        ref_windows = window_partition(shifted_ref, self.window_size)
        ref_windows = ref_windows.view(-1, self.window_size * self.window_size, C)

        # W-MCA/SW-MCA (窗口/移位窗口交叉注意力)
        # if self.shift_size > 0:
        #     print("x_windows shape:", x_windows.shape)
        #     print("ref_windows shape:", ref_windows.shape)
        attn_windows = self.attn(x_windows, ref_windows, mask=self.attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # 确保 window_reverse 函数已定义
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 逆向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        return x


# window size 8
class CrossHSA(nn.Module):
    def __init__(
            self,
            dim,
            stage=1,
    ):
        super().__init__()
        self.wa = CrossSwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=0)
        self.swa = CrossSwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=4)
        self.pn = PreNorm(dim, FeedForward(dim=dim))

    def forward(self, x, ref):
        """
        x: [b,h,w,c]
        return out: [b,c,h,w]
        """
        x = self.wa(x, ref) + x
        x = self.swa(x, ref) + x
        x = self.pn(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class MA(nn.Module):             
    def __init__(
            self, n_feat):
        super(MA, self).__init__()
        self.depth_conv = nn.Conv2d(n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat)

    def forward(self, mask_3d):
        attn_map = torch.sigmoid(self.depth_conv(mask_3d))
        res =  mask_3d * attn_map
        mask_attn = res + mask_3d
        return mask_attn

class CrossSSM_AB(nn.Module):        
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            attention_type = 'full'
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.ma = MA(dim)        # Mask Attention (MA)
        self.sa = CrossHSA(dim)       # Cross Hierarchical spatial attention (HSA)
        self.sa_conv = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.dim = dim
        self.attention_type = attention_type

    def forward(self, x_in, spa_g=None, ref=None):
        """
        x_in: [b,h,w,c]
        ref: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape

        ref_attn = self.ma(ref.permute(0,3,1,2)).permute(0,2,3,1)
        if b != 0:
            ref_attn = (ref_attn[0, :, :, :]).expand([b, h, w, c])
        
        ## no spatial attention in baseline
        if self.attention_type == 'base':
            x = x_in.reshape(b,h*w,c)
        else:   
            x_mid_out=self.sa(x_in, ref_attn)
            x_sa_emb=self.sa_conv(spa_g)+spa_g
            if x_sa_emb.shape[3] != x_mid_out.shape[3]:   
                x_sa_emb = shift_back(x_sa_emb)
            x = (x_mid_out*x_sa_emb).permute(0,2,3,1).reshape(b,h*w,c)
            
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # ref_attn = self.ma(ref.permute(0,3,1,2)).permute(0,2,3,1)
        # if b != 0:
        #     ref_attn = (ref_attn[0, :, :, :]).expand([b, h, w, c])
        q, k, v, ref_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp, ref_attn.flatten(1, 2)))
        
        ## completed CAB
        if self.attention_type == 'full':
            v = v * ref_attn
    
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
        return out

class CrossCAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=1,
            attention_type = 'full'
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                CrossSSM_AB(dim=dim, dim_head=dim_head, heads=heads,attention_type = attention_type),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, ref):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, ref, ref.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



class ModulatedSpatialSpectralFusion(nn.Module):
    """
    一个受 S3Conv 和 SM-FFN 启发的调制融合模块 (MSSFusion)。
    该模块通过并行的"值"(Value)和"门"(Gate)分支来融合两个特征。
    每个分支内部都采用了空间(深度可分离)和光谱(逐点)卷积的分离思想，
    以实现高效且强大的特征交互。
    """
    def __init__(self, c_in):
        # c_in 即为 dim_stage // 2
        super().__init__()

        # 1. Spatial Processing Sub-Branch
        self.value_spatial = nn.Sequential(
            # Use depthwise convolution to process spatial info from the concatenated features
            nn.Conv2d(c_in * 2, c_in * 2, kernel_size=3, padding=1, groups=c_in * 2, bias=False),
            nn.LeakyReLU(),
            # Use a 1x1 conv to project back to the target channel dimension
            nn.Conv2d(c_in * 2, c_in, kernel_size=1, bias=False),
            nn.LeakyReLU()
        )
        
        # 2. Spectral Processing Sub-Branch
        self.value_spectral = nn.Sequential(
            # Use 1x1 convolution to mix channel information
            nn.Conv2d(c_in * 2, c_in, kernel_size=1, bias=False),
        )

        self.spatial_attention_head = nn.Sequential(
            nn.Conv2d(c_in * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.spectral_attention_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in * 2, c_in, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, fea, fea_en):
        B, C, H, W = fea.shape

        if fea_en.shape[2:] != (H, W):
            fea_en = F.interpolate(fea_en, size=(H, W), mode="bilinear", align_corners=False)

        # 1. 合并输入特征
        x = torch.cat([fea, fea_en], dim=1) # shape: (B, c_in * 2, H, W)

        # 2. 通过两个并行分支处理
        # value = self.value_branch(x)   # "值"
        # 2. Process through the parallel Value branch
        value_s = self.value_spatial(x)    # Spatial path
        value_p = self.value_spectral(x)  # Spectral (pointwise) path

        # gate_logits = self.gate_branch(x) # "门"的原始输出
        w_spatial = self.spatial_attention_head(x)   # Shape: (B, 1, H, W)
        w_spectral = self.spectral_attention_head(x) # Shape: (B, c_in, 1, 1)

        # 3. 计算门控融合结果
        # "门"分支通过Sigmoid函数生成0到1之间的空间权重
        fused_out = (value_s * w_spatial) + (value_p * w_spectral)

        # 4. 添加残差连接
        output = fused_out + fea
        
        return output

class UAFL(nn.Module):
    def __init__(self, dim=28, stage=1, num_blocks=[2,1],attention_type = 'full',numend=3):
        super(UAFL, self).__init__()
        self.dim = dim
        self.stage = stage
        self.numend = numend

        # Input projection
        self.embedding = nn.Conv2d(numend, self.dim, 3, 1, 1, bias=False)
        self.embedding2 = nn.Conv2d(3, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                CrossCAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim,attention_type = attention_type),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                DCN_Refine_With_Prior_flow(dim_stage, dim_stage, kernel_size=(3,3), stride=1, padding=1, deformable_groups=8)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = CrossCAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1],attention_type = attention_type)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                ModulatedSpatialSpectralFusion(dim_stage // 2),
                CrossCAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim,attention_type = attention_type),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, numend, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, ref=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # if ref == None:
        #     ref = self.lrelu(self.embedding2(x))

        A_hat, E_y = Unmix_svd_3d(x, self.numend)  # A_hat: b,numend,h,w; E_y: b,r,c

        # Embedding
        fea = self.lrelu(self.embedding(A_hat))
        ref_fea = self.lrelu(self.embedding2(ref))

        # Encoder
        fea_encoder = []
        refs = []
        for (CrossCAB, FeaDownSample, RefDownSample, RefConv) in self.encoder_layers:
            ref_fea = RefConv(ref_fea, fea)
            fea = CrossCAB(fea, ref_fea)
            refs.append(ref_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            ref_fea = RefDownSample(ref_fea)

        # Bottleneck
        fea = self.bottleneck(fea, ref_fea)

        # Decoder
        for i, (FeaUpSample, Fution, CrossCAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            # fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = Fution(fea, fea_encoder[self.stage-1-i] )
            ref_fea = refs[self.stage - 1 - i]
            fea = CrossCAB(fea, ref_fea)

        # Mapping
        # out = self.mapping(fea) + x
        out = mix(self.mapping(fea), E_y) + x

        return out