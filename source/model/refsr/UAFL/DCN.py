#!/usr/bin/env python

import logging
import math

#import _ext as _backend
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import pdb
import torch.utils.checkpoint as checkpoint
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d


logger = logging.getLogger('base')

"""
class _DCNv2(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias, stride, padding,
                dilation, deformable_groups):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.deformable_groups = deformable_groups
        output = _backend.dcn_v2_forward(
            input, weight, bias, offset, mask, ctx.kernel_size[0],
            ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0],
            ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
            ctx.deformable_groups)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            _backend.dcn_v2_backward(input, weight,
                                     bias,
                                     offset, mask,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.deformable_groups)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None,


dcn_v2_conv = _DCNv2.apply
"""

class DCNv2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[
            0] * self.kernel_size[1] == offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[
            1] == mask.shape[1]
        return modulated_deform_conv2d(input, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, 1,
                           self.deformable_groups)


class DCN(DCNv2):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(input, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, 1,
                           self.deformable_groups)


class DCN_sep(DCNv2):
    '''Use other features to generate offsets and masks'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1,
                 extra_offset_mask=True):
        super(DCN_sep,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, deformable_groups)
        self.extra_offset_mask = extra_offset_mask
        channels_ = self.deformable_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        if self.extra_offset_mask:
            # x = [input, features]
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            logger.warning(
                'Offset mean is {}, larger than 100.'.format(offset_mean))
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation, 1,
                           self.deformable_groups)
    




import torch.nn.functional as F

# 步骤 3: 从小数部分生成位置编码 (PE)
def generate_pe(decimal_offset, num_freqs=8, temperature=10000.0):
    """
    Generate sinusoidal positional encoding from decimal offsets.
    Args:
        decimal_offset: Tensor of shape [B, 2, H, W] with values in [-0.5, 0.5].
        num_freqs: Number of frequency bands (L in many papers).
    Returns:
        pe: Tensor of shape [B, 2 * 2 * num_freqs, H, W].
    """
    # [B, 2, H, W] -> [B, H, W, 2]
    decimal_offset = decimal_offset.permute(0, 2, 3, 1) 
    
    device = decimal_offset.device
    dtype = decimal_offset.dtype
    
    freq_bands = temperature ** (torch.arange(num_freqs, device=device, dtype=dtype) / num_freqs)
    # [1, 1, 1, 1, num_freqs]
    freq_bands = freq_bands.view(1, 1, 1, 1, -1)
    
    # [B, H, W, 2, 1] * [1, 1, 1, 1, num_freqs] -> [B, H, W, 2, num_freqs]
    inputs = decimal_offset.unsqueeze(-1) * freq_bands
    
    # [B, H, W, 2 * num_freqs]
    embedded = torch.cat([torch.sin(inputs), torch.cos(inputs)], dim=-1)
    # [B, H, W, 2 * 2 * num_freqs]
    pe = embedded.flatten(-2)
    
    # [B, 2 * 2 * num_freqs, H, W]
    return pe.permute(0, 3, 1, 2)

# 步骤 4: Warp/resample a feature map using grid_sample
def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
        """Warp an image or feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2), normal value.
            interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
            padding_mode (str): 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Before pytorch 1.3, the default value is
                align_corners=True. After pytorch 1.3, the default value is
                align_corners=False. Here, we use the True as default.
        Returns:
            Tensor: Warped image or feature map.
        """

        flow = flow.permute(0, 2, 3, 1)
        assert x.size()[-2:] == flow.size()[1:3]
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).type_as(x),
            torch.arange(0, w).type_as(x))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False

        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x,
                               vgrid_scaled,
                               mode=interp_mode,
                               padding_mode=padding_mode,
                               align_corners=align_corners)

        return output

    

class PyramidFlowPredictor(nn.Module):
    """
    一个两层的轻量级金字塔网络，用于预测单点的 pre_flow 和 pre_sim。
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # --- [修改] 输出通道数现在是 2 (flow) 和 1 (sim) ---
        self.num_flow_channels = 2
        self.num_sim_channels = 1 

        # Level 1 (Coarse): 简化为单层卷积，只预测 flow
        self.level1_predictor = nn.Conv2d(
            in_channels * 2, 
            self.num_flow_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )

        # Level 0 (Fine): 简化为单层卷积，预测残差flow和similarity
        self.level0_predictor = nn.Conv2d(
            in_channels * 2, 
            self.num_flow_channels + self.num_sim_channels,
            kernel_size=3, 
            stride=1, 
            padding=1
        )

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.init_weights()

    def init_weights(self):
        self.level1_predictor.weight.data.zero_()
        self.level1_predictor.bias.data.zero_()
        self.level0_predictor.weight.data.zero_()
        self.level0_predictor.bias.data.zero_()

    def forward(self, ref_feat, target_feat):
        # Level 1: Coarse Prediction
        ref_feat_l1 = self.downsample(ref_feat)
        target_feat_l1 = self.downsample(target_feat)
        concatenated_l1 = torch.cat([ref_feat_l1, target_feat_l1], dim=1)
        flow_l1 = self.level1_predictor(concatenated_l1) * 2.0
        
        # Upsample coarse flow to original resolution
        flow_l0_from_l1 = F.interpolate(flow_l1, scale_factor=2, mode='bilinear', align_corners=False)

        # Warp original ref_feat with the upsampled coarse flow
        warped_ref_feat_l0 = flow_warp(ref_feat, flow_l0_from_l1)

        # Level 0: Fine-grained Prediction (predicts residual)
        concatenated_l0 = torch.cat([warped_ref_feat_l0, target_feat], dim=1)
        residual_out = self.level0_predictor(concatenated_l0)
        
        residual_flow = residual_out[:, :self.num_flow_channels, :, :]
        pre_sim_logits = residual_out[:, self.num_flow_channels:, :, :]
        
        # Combine coarse and residual flows
        final_flow = flow_l0_from_l1 + residual_flow
        
        # Apply sigmoid to similarity
        pre_sim = torch.sigmoid(pre_sim_logits)

        return final_flow, pre_sim
    
class DCN_Refine_With_Prior_flow(DCNv2):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1,
                 dilation=1,
                 deformable_groups=1,
                 max_residue_magnitude=10):
        
        super(DCN_Refine_With_Prior_flow, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        
        print('DCN_Refine_With_Prior_flow (Flow-based) Used!!!')
        self.max_residue_magnitude = max_residue_magnitude

        # --- [修改] 内部创建我们的 Flow 预测器 ---
        self.prior_predictor = PyramidFlowPredictor(in_channels)

        # Refinement Network 保持不变
        num_pe_channels = 2 * 2 * 8
        refine_in_channels = in_channels + in_channels + num_pe_channels
        dcn_offset_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        dcn_mask_channels = self.deformable_groups * 1 * self.kernel_size[0] * self.kernel_size[1]
        refine_out_channels = dcn_offset_channels + dcn_mask_channels
        self.refinement_net = nn.Sequential(
            nn.Conv2d(refine_in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, refine_out_channels, 3, 1, 1)
        )
        
        self.init_weights()

    def init_weights(self):
        self.refinement_net[-1].weight.data.zero_()
        self.refinement_net[-1].bias.data.zero_()

    def forward(self, ref_feat, target_feat):
        # --- 1. 使用内部预测器生成先验 Flow 和 Sim ---
        # pre_flow: [b, 2, h, w], pre_sim: [b, 1, h, w]
        pre_flow, pre_sim = self.prior_predictor(ref_feat, target_feat)

        # --- 2. Warp & PE Generation (直接使用 pre_flow) ---
        flow_for_warp = pre_flow
        offset_decimal = flow_for_warp - torch.floor(flow_for_warp)
        pe_decimal = generate_pe(offset_decimal)
        warped_ref_feat = flow_warp(ref_feat, flow_for_warp)

        # --- 3. 预测残差 ---
        refinement_input = torch.cat([target_feat, warped_ref_feat, pe_decimal], dim=1)
        refinement_out = self.refinement_net(refinement_input)

        # --- 4. 组合 Priors 和 Residuals ---
        # a. 拆分
        dcn_offset_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        residual_offset = refinement_out[:, :dcn_offset_channels, :, :]
        residual_mask_logits = refinement_out[:, dcn_offset_channels:, :, :]
        
        # b. 缩放 residual_offset
        if self.max_residue_magnitude:
            residual_offset = self.max_residue_magnitude * torch.tanh(residual_offset)

        # --- c. [修改] 组合 Offset ---
        # 将单点 pre_flow [b, 2, h, w] 扩展为 DCN 需要的 pre_offset [b, G*18, h, w]
        k_sq = self.kernel_size[0] * self.kernel_size[1]
        # [b, 2, h, w] -> unsqueeze -> [b, 1, 2, h, w] -> repeat -> [b, 9, 2, h, w]
        pre_offset_expanded = pre_flow.unsqueeze(1).repeat(1, k_sq, 1, 1, 1)
        # -> reshape -> [b, 18, h, w]
        b, _, _, h, w = pre_offset_expanded.size()
        pre_offset_interleaved = pre_offset_expanded.view(b, -1, h, w)
        # -> repeat -> [b, 144, h, w]
        pre_offset_reordered = pre_offset_interleaved.repeat(1, self.deformable_groups, 1, 1)
        
        final_offset = pre_offset_reordered + residual_offset

        # --- d. [修改] 组合 Mask ---
        # pre_sim [b, 1, h, w] 需要被 repeat 以匹配 G*K*K 的通道
        pre_sim_repeated = pre_sim.repeat(1, self.deformable_groups * k_sq, 1, 1)
        final_mask = torch.sigmoid(residual_mask_logits * pre_sim_repeated)

        # --- 5. Final DCN Call ---
        return modulated_deform_conv2d(ref_feat, final_offset, final_mask, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation, 1,
                                       self.deformable_groups)