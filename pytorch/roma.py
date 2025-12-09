# -*- coding: utf-8 -*-
# @Author  : Parskatt

import math
import torch
import warnings
import torch.nn.functional as F
import torchvision.models as tvm

from torch import nn
from einops import rearrange

# noinspection PyPackages
from dino import vit_large, Block,Attention



resolutions = {
    "low": (448, 448),
    "medium": (14 * 8 * 5, 14 * 8 * 5),
    "high": (14 * 8 * 6, 14 * 8 * 6),
}

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

class GP16(nn.Module):
    
    #最终优化版：
    #1. 通过手动实现高效的余弦相似度，解决了推理速度慢的问题。
    #2. 保留了可调的温度参数，以确保数值近似度。
    #3. 100% ONNX 兼容且性能极高。
    
    def __init__(self, kernel, gp_dim=512, sigma_noise=0.1):
        super().__init__()
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        
        # 预计算坐标网格
        h, w = 36, 36
        coarse_coords_h = torch.linspace(-1 + 1 / h, 1 - 1 / h, h)
        coarse_coords_w = torch.linspace(-1 + 1 / w, 1 - 1 / w, w)
        grid_y, grid_x = torch.meshgrid(coarse_coords_h, coarse_coords_w, indexing="ij")
        coarse_coords = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).permute(0, 3, 1, 2)
        self.register_buffer("coarse_grid", coarse_coords)
        
        # 关键的温度参数，根据您的测试设为 0.1
        self.temperature = 0.1
        self.eps = 1e-6 # 用于数值稳定的小常数

    def project_to_basis(self, x):
        return torch.cos(8 * math.pi * self.pos_conv(x))

    def get_pos_enc(self, y):
        b, _, _, _ = y.shape
        coarse_coords = self.coarse_grid.expand(b, -1, -1, -1)
        return self.project_to_basis(coarse_coords)

    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        _, _, h2, w2 = y.shape
        hw1, hw2 = h1 * w1, h2 * w2

        f = self.get_pos_enc(y)
        b, d, _, _ = f.shape

        q = x.float().view(b, c, hw1).permute(0, 2, 1) # Query: [B, hw1, C]
        k = y.float().view(b, c, hw2).permute(0, 2, 1) # Key:   [B, hw2, C]
        v = f.view(b, d, hw2).permute(0, 2, 1)       # Value: [B, hw2, D]

        # --- 高效的、手动的余弦相似度计算 ---
        # 1. L2 归一化
        q_norm = F.normalize(q, p=2, dim=-1, eps=self.eps)
        k_norm = F.normalize(k, p=2, dim=-1, eps=self.eps)

        # 2. 使用 matmul 计算点积，这等价于归一化向量的余弦相似度
        # (B, hw1, C) @ (B, C, hw2) -> (B, hw1, hw2)
        attn_scores = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        # --- 优化结束 ---
        
        # 应用温度并进行 softmax 归一化
        attn_probs = (attn_scores / self.temperature).softmax(dim=-1)

        # 计算加权和
        mu_x = torch.matmul(attn_probs, v) # [B, hw1, D]

        # Reshape 回 4D 格式
        gp_posterior = mu_x.permute(0, 2, 1).view(b, d, h1, w1)
        
        return gp_posterior




class VGG19(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])

    def forward(self, x):
        feat_s1 = self.layers[5](self.layers[4](self.layers[3](self.layers[2](self.layers[1](self.layers[0](x))))))
        
        # Pass through the first MaxPool (layer 6) to get the input for the next block
        x_after_pool1 = self.layers[6](feat_s1)

        # --- Block 2 (produces feat_s2) ---
        # Pass through layers 7 to 12
        feat_s2_intermediate = x_after_pool1
        for i in range(7, 13):
            feat_s2_intermediate = self.layers[i](feat_s2_intermediate)
        feat_s2 = feat_s2_intermediate # Capture before the second MaxPool

        # Pass through the second MaxPool (layer 13)
        x_after_pool2 = self.layers[13](feat_s2)

        # --- Block 3 (produces feat_s4) ---
        feat_s4_intermediate = x_after_pool2
        for i in range(14, 26):
            feat_s4_intermediate = self.layers[i](feat_s4_intermediate)
        feat_s4 = feat_s4_intermediate # Capture before the third MaxPool

        # Pass through the third MaxPool (layer 26)
        x_after_pool3 = self.layers[26](feat_s4)

        # --- Block 4 (produces feat_s8) ---
        feat_s8_intermediate = x_after_pool3
        for i in range(27, 39):
            feat_s8_intermediate = self.layers[i](feat_s8_intermediate)
        feat_s8 = feat_s8_intermediate # Capture before the fourth MaxPool
        
        # The final MaxPool (layer 39) is not used for feature extraction in this model
        
        return feat_s1, feat_s2, feat_s4, feat_s8


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_decoder,
        gps,
        proj,
        conv_refiner,
        detach=False,
        scales="all",
        pos_embeddings=None,
        num_refinement_steps_per_scale=1,
        warp_noise_std=0.0,
        displacement_dropout_p=0.0,
        gm_warp_dropout_p=0.0,
        flow_upsample_mode="bilinear",
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        self.pos_embeddings = {}
        self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode

    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords

    def get_positional_embedding(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords
    
    def forward(self, f1_s1, f1_s2, f1_s4, f1_s8, f1_s16,
                               f2_s1, f2_s2, f2_s4, f2_s8, f2_s16):
        
        b = f1_s1.shape[0]
        device = f1_s1.device
        
        # 注意：sizes_X 的获取代码其实可以删掉了，因为我们不再依赖它来做插值
        sizes_1 = f1_s1.shape[-2:] 

        # --- Scale 16 ---
        f1_s, f2_s = self.proj["16"](f1_s16), self.proj["16"](f2_s16)
        gp_posterior = self.gps["16"](f1_s, f2_s)
        gm_warp_or_cls, certainty16 = self.embedding_decoder(gp_posterior, f1_s)
        flow16 = cls_to_flow_refine(gm_warp_or_cls).permute(0, 3, 1, 2)
        
        # --- Scale 8 ---
        # 修改点：使用 scale_factor=2
        flow_up_to_8 = F.interpolate(flow16, scale_factor=1.75, mode=self.flow_upsample_mode, align_corners=False)
        cert_up_to_8 = F.interpolate(certainty16, scale_factor=1.75, mode=self.flow_upsample_mode, align_corners=False)
        
        f1_s, f2_s = self.proj["8"](f1_s8), self.proj["8"](f2_s8)
        delta_flow, delta_cert = self.conv_refiner["8"](f1_s, f2_s, flow_up_to_8, logits=cert_up_to_8)
        # 注意：这里除以 sizes_1 (原图尺寸) 是必要的归一化步骤，保留即可
        flow8 = flow_up_to_8 + 8 * torch.stack((delta_flow[:, 0] / (self.refine_init * sizes_1[1]), delta_flow[:, 1] / (self.refine_init * sizes_1[0])), dim=1)
        certainty8 = cert_up_to_8 + delta_cert

        # --- Scale 4 ---
        # 修改点：使用 scale_factor=2
        flow_up_to_4 = F.interpolate(flow8, scale_factor=2, mode=self.flow_upsample_mode, align_corners=False)
        cert_up_to_4 = F.interpolate(certainty8, scale_factor=2, mode=self.flow_upsample_mode, align_corners=False)

        f1_s, f2_s = self.proj["4"](f1_s4), self.proj["4"](f2_s4)
        delta_flow, delta_cert = self.conv_refiner["4"](f1_s, f2_s, flow_up_to_4, logits=cert_up_to_4)
        flow4 = flow_up_to_4 + 4 * torch.stack((delta_flow[:, 0] / (self.refine_init * sizes_1[1]), delta_flow[:, 1] / (self.refine_init * sizes_1[0])), dim=1)
        certainty4 = cert_up_to_4 + delta_cert

        # --- Scale 2 ---
        # 修改点：使用 scale_factor=2
        flow_up_to_2 = F.interpolate(flow4, scale_factor=2, mode=self.flow_upsample_mode, align_corners=False)
        cert_up_to_2 = F.interpolate(certainty4, scale_factor=2, mode=self.flow_upsample_mode, align_corners=False)

        f1_s, f2_s = self.proj["2"](f1_s2), self.proj["2"](f2_s2)
        delta_flow, delta_cert = self.conv_refiner["2"](f1_s, f2_s, flow_up_to_2, logits=cert_up_to_2)
        flow2 = flow_up_to_2 + 2 * torch.stack((delta_flow[:, 0] / (self.refine_init * sizes_1[1]), delta_flow[:, 1] / (self.refine_init * sizes_1[0])), dim=1)
        certainty2 = cert_up_to_2 + delta_cert

        # --- Scale 1 ---
        # 修改点：使用 scale_factor=2
        flow_up_to_1 = F.interpolate(flow2, scale_factor=2, mode=self.flow_upsample_mode, align_corners=False)
        cert_up_to_1 = F.interpolate(certainty2, scale_factor=2, mode=self.flow_upsample_mode, align_corners=False)

        f1_s, f2_s = self.proj["1"](f1_s1), self.proj["1"](f2_s1)
        delta_flow, delta_cert = self.conv_refiner["1"](f1_s, f2_s, flow_up_to_1, logits=cert_up_to_1)
        flow1 = flow_up_to_1 + 1 * torch.stack((delta_flow[:, 0] / (self.refine_init * sizes_1[1]), delta_flow[:, 1] / (self.refine_init * sizes_1[0])), dim=1)
        certainty1 = cert_up_to_1 + delta_cert

        return flow1, certainty1, certainty16

    

class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K




class ConvRefinerWithCorr(nn.Module):
    """
    Specialized ConvRefiner for scales that USE local correlation (16, 8, 4).
    This class has a static computation graph with no conditional branches.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dw=False, kernel_size=5, hidden_blocks=3,
                 displacement_emb_dim=None, local_corr_radius=None, 
                 corr_in_other=True, disable_local_corr_grad=False, bn_momentum=0.1, **kwargs):
        # We accept **kwargs to ignore unused parameters from the original call like `displacement_emb`.
        super().__init__()
        self.bn_momentum = bn_momentum
        # The first block's input dimension is fixed based on the forward pass logic
        actual_in_dim = in_dim 
        self.block1 = self.create_block(actual_in_dim, hidden_dim, dw=dw, kernel_size=kernel_size)
        self.hidden_blocks = nn.Sequential(
            *[self.create_block(hidden_dim, hidden_dim, dw=dw, kernel_size=kernel_size) for _ in range(hidden_blocks)]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        
        # These layers are guaranteed to exist for this class
        self.disp_emb = nn.Conv2d(2, displacement_emb_dim, 1, 1, 0)
        self.local_corr_radius = local_corr_radius
        self.sample_mode = "bilinear" # Hardcoded from original defaults
    
    def fuse(self):
        """
        在导出前调用此方法。
        它会将 Conv 和 BN 的权重合并，并将 BN 替换为 Identity。
        """
        def fuse_conv_bn(conv, bn):
            # 获取 Conv 和 BN 的参数
            w = conv.weight
            mean = bn.running_mean
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            beta = bn.bias
            
            if conv.bias is not None:
                b = conv.bias
            else:
                b = mean.new_zeros(mean.shape)

            # 计算融合后的权重和偏置
            # W_new = W * (gamma / std)
            w_new = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
            
            # b_new = (b - mean) * (gamma / std) + beta
            b_new = (b - mean) * (gamma / var_sqrt) + beta
            
            # 创建新的卷积层
            fused_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels,
                conv.kernel_size, conv.stride, conv.padding,
                conv.dilation, conv.groups, bias=True
            )
            
            # 赋值参数
            fused_conv.weight.data = w_new
            fused_conv.bias.data = b_new
            fused_conv.to(w.device)
            return fused_conv

        # 1. 融合 self.block1
        # 结构是 Sequential(conv1, norm, relu, conv2)
        # 索引：0=Conv1, 1=BN
        if isinstance(self.block1[1], nn.BatchNorm2d):
            fused_c1 = fuse_conv_bn(self.block1[0], self.block1[1])
            self.block1[0] = fused_c1
            self.block1[1] = nn.Identity() # 替换 BN 为直通

        # 2. 融合 self.hidden_blocks (这是一个 Sequential 的 Sequential)
        for block in self.hidden_blocks:
            if isinstance(block[1], nn.BatchNorm2d):
                fused_c = fuse_conv_bn(block[0], block[1])
                block[0] = fused_c
                block[1] = nn.Identity()

    def create_block(self, in_dim, out_dim, dw=False, kernel_size=5, bias=True, norm_type=nn.BatchNorm2d):
        num_groups = in_dim if dw else 1
        if dw:
            assert out_dim % in_dim == 0, "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(in_dim, out_dim, kernel_size, 1, kernel_size // 2, groups=num_groups, bias=bias)
        norm = norm_type(out_dim, momentum=self.bn_momentum)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
    
    
    
    def forward(self, x, y, flow, scale_factor=1.0, logits=None):
        b, c, hs, ws = x.shape
        
        # The `with torch.no_grad():` block is for training. During ONNX export or inference,
        # it's redundant but harmless. We keep it to match the original structure.
        with torch.no_grad():
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1).to(torch.float32), align_corners=False, mode=self.sample_mode)
        
        # Branch `if self.has_displacement_emb:` is removed, as it's always True.
        im_A_coords_y = torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device)
        im_A_coords_x = torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device)
        grid_y, grid_x = torch.meshgrid(im_A_coords_y, im_A_coords_x, indexing='ij')
        im_A_coords = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1).permute(0, 3, 1, 2)

        in_displacement = flow - im_A_coords
        emb_in_displacement = self.disp_emb(40.0 / 32.0 * scale_factor * in_displacement)
        
        # Branch `if self.local_corr_radius:` is removed, as it's always True for this class.
        # Branch `if self.corr_in_other:` is also removed.
        local_corr = local_correlation(x, y, local_radius=self.local_corr_radius, flow=flow, sample_mode=self.sample_mode)
        
        d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
        
        # 注意：由于 BN 变成了 Identity，这里的执行逻辑完全不受影响，但速度更快
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty


class ConvRefinerNoCorr(nn.Module):
    """
    Specialized ConvRefiner for scales that do NOT use local correlation (2, 1).
    This class has a static computation graph with no conditional branches.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dw=False, kernel_size=5, hidden_blocks=3,
                 displacement_emb_dim=None, bn_momentum=0.1, **kwargs):
        # We accept **kwargs to ignore unused parameters like `local_corr_radius=None`.
        super().__init__()
        self.bn_momentum = bn_momentum
        actual_in_dim = in_dim
        self.block1 = self.create_block(actual_in_dim, hidden_dim, dw=dw, kernel_size=kernel_size)
        self.hidden_blocks = nn.Sequential(
            *[self.create_block(hidden_dim, hidden_dim, dw=dw, kernel_size=kernel_size) for _ in range(hidden_blocks)]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)

        # This layer is guaranteed to exist for this class
        self.disp_emb = nn.Conv2d(2, displacement_emb_dim, 1, 1, 0)
        self.sample_mode = "bilinear"
   
    def fuse(self):
        """
        将 Block 中的 Conv 和 BN 融合，并将 BN 替换为 Identity。
        必须在 model.eval() 模式下调用。
        """
        def fuse_conv_bn(conv, bn):
            # 1. 获取参数
            w = conv.weight
            mean = bn.running_mean
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            beta = bn.bias
            
            if conv.bias is not None:
                b = conv.bias
            else:
                b = mean.new_zeros(mean.shape)

            # 2. 计算融合后的权重和偏置
            # W_new = W * (gamma / std)
            w_new = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
            
            # b_new = (b - mean) * (gamma / std) + beta
            b_new = (b - mean) * (gamma / var_sqrt) + beta
            
            # 3. 创建新的卷积层
            fused_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels,
                conv.kernel_size, conv.stride, conv.padding,
                conv.dilation, conv.groups, bias=True
            )
            
            # 4. 赋值参数
            fused_conv.weight.data = w_new
            fused_conv.bias.data = b_new
            fused_conv.to(w.device)
            return fused_conv

        # 1. 融合 self.block1
        # 结构: [0:Conv, 1:BN, 2:ReLU, 3:Conv]
        if isinstance(self.block1[1], nn.BatchNorm2d):
            fused_c1 = fuse_conv_bn(self.block1[0], self.block1[1])
            self.block1[0] = fused_c1
            self.block1[1] = nn.Identity()

        # 2. 融合 self.hidden_blocks
        for block in self.hidden_blocks:
            if isinstance(block[1], nn.BatchNorm2d):
                fused_c = fuse_conv_bn(block[0], block[1])
                block[0] = fused_c
                block[1] = nn.Identity()

    def create_block(self, in_dim, out_dim, dw=False, kernel_size=5, bias=True, norm_type=nn.BatchNorm2d):
        num_groups = in_dim if dw else 1
        if dw:
            assert out_dim % in_dim == 0, "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(in_dim, out_dim, kernel_size, 1, kernel_size // 2, groups=num_groups, bias=bias)
        norm = norm_type(out_dim, momentum=self.bn_momentum)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
    

    def forward(self, x, y, flow, scale_factor=1.0, logits=None):
        b, c, hs, ws = x.shape

        with torch.no_grad():
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1).to(torch.float32), align_corners=False, mode=self.sample_mode)

        # Branch `if self.has_displacement_emb:` is removed.
        im_A_coords_y = torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device)
        im_A_coords_x = torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device)
        grid_y, grid_x = torch.meshgrid(im_A_coords_y, im_A_coords_x, indexing='ij')
        im_A_coords = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1).permute(0, 3, 1, 2)

        in_displacement = flow - im_A_coords
        emb_in_displacement = self.disp_emb(40.0 / 32.0 * scale_factor * in_displacement)
        
        # Branch `if self.local_corr_radius:` is removed, as it's never entered for this class.
        
        d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
        
        # Branch `if self.concat_logits:` is removed.
        
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty


class CNNandDinov2(nn.Module):
    def __init__(
        self,
        dinov2_weights=None,
        device="cuda"
    ):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                map_location="cpu",
            )

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )

   
        
        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        self.cnn = VGG19()
        self.dinov2_vitl14 = [dinov2_vitl14]
        #self.dinov2_vitl14 = nn.ModuleList([dinov2_vitl14])  # ugly hack to not show parameters to DDP


    def train(self, mode: bool = True):
        return self.cnn.train(mode)

    def forward(self, x):
        """
        A modified forward pass that returns a tuple of tensors in a fixed order,
        suitable for ONNX export.
        """
        # The computation is identical to the original forward pass.
        B, C, H, W = x.shape
        #feature_pyramid = self.cnn(x)
        vgg_features_tuple = self.cnn.forward(x)
        # Note: DINOv2 part is under torch.no_grad() context in the original code,
        # which is good practice but not strictly necessary for the tracer.
        # We will keep it for consistency.
        with torch.no_grad():
            if self.dinov2_vitl14[0].device != x.device:
                self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device)
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x)
            features_16 = (
                dinov2_features_16["x_norm_patchtokens"]
                .permute(0, 2, 1)
                .reshape(B, 1024, H // 14, W // 14)
            )
            # del is a Python operation, not needed here as we return directly
        
        #feature_pyramid[16] = features_16
        
        # Return a tuple with tensors in a predefined, fixed order.
        # This order must be documented and used when re-creating the dictionary.
            return (
                vgg_features_tuple[0], # feat_s1
                vgg_features_tuple[1], # feat_s2
                vgg_features_tuple[2], # feat_s4
                vgg_features_tuple[3], # feat_s8
                features_16,
            )


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=504,
        w=504,
        sample_mode="threshold_balanced",
        upsample_preds=False,
        symmetric=True,
        name=None,
        attenuate_cert=True,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = (14 * 16 * 6, 14 * 16 * 6)
        self.symmetric = symmetric
        self.sample_thresh = 0.05

    def get_output_resolution(self):
        return self.h_resized, self.w_resized
        

    def extract_backbone_features(self, batch, batched=True, upsample=False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        X = torch.cat((x_q, x_s), dim=0)
        feature_pyramid = self.encoder(X)
        return feature_pyramid

    def sample(
            self,
            dense_matches,
            dense_certainty,
            num=5000,
    ):
        
        upper_thresh = self.sample_thresh
        dense_certainty = dense_certainty.clone()
        dense_certainty_ = dense_certainty.clone()
        dense_certainty[dense_certainty > upper_thresh] = 1
        matches, certainty = (
            dense_matches.reshape(-1, 4),
            dense_certainty.reshape(-1),
        )
        # noinspection PyUnboundLocalVariable
        certainty_ = dense_certainty_.reshape(-1)
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        if not certainty.sum(): certainty = certainty + 1e-8
        good_samples = torch.multinomial(certainty,
                                         num_samples=min(expansion_factor * num, len(certainty)),
                                         replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        good_certainty_ = certainty_[good_samples]
        good_certainty = good_certainty_
        

        density = kde(good_matches, std=0.1)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p,
                                             num_samples = min(num,len(good_certainty)),
                                             replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]


       
    def forward_symmetric(self, batch, batched=True, upsample=False, scale_factor=1):
        feature_pyramid = self.extract_backbone_features(
            batch, batched=batched, upsample=upsample
        )
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim=0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            scale_factor=scale_factor,
        )
        return corresps

    # --- 这是我们为 ONNX 准备的静态方法 ---
    def forward_symmetric_for_onnx(self, image_a, image_b):
        X = torch.cat((image_a, image_b), dim=0)
        
        # 假设 encoder 有 forward_for_onnx
        feature_pyramid_tuple = self.encoder.forward(X)
        
        f_q_pyramid_tuple = feature_pyramid_tuple
        f_s_pyramid_tuple = tuple(
            [torch.cat((f.chunk(2)[1], f.chunk(2)[0]), dim=0) for f in feature_pyramid_tuple]
        )
        
        decoder_inputs = (*f_q_pyramid_tuple, *f_s_pyramid_tuple)
        
        # 假设 decoder 有 forward_for_onnx
        final_flow, final_certainty, low_res_certainty = self.decoder.forward(*decoder_inputs)
        
        return final_flow, final_certainty, low_res_certainty

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        kpts_A = torch.stack(
            (W_A / 2 * (kpts_A[..., 0] + 1), H_A / 2 * (kpts_A[..., 1] + 1)), dim=-1
        )
        kpts_B = torch.stack(
            (W_B / 2 * (kpts_B[..., 0] + 1), H_B / 2 * (kpts_B[..., 1] + 1)), dim=-1
        )
        return kpts_A, kpts_B

    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        kpts_A = torch.stack(
            (2 / W_A * kpts_A[..., 0] - 1, 2 / H_A * kpts_A[..., 1] - 1), dim=-1
        )
        kpts_B = torch.stack(
            (2 / W_B * kpts_B[..., 0] - 1, 2 / H_B * kpts_B[..., 1] - 1), dim=-1
        )
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple=True, return_inds=False):
        x_A_to_B = F.grid_sample(
            warp[..., -2:].permute(2, 0, 1)[None],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0].mT
        cert_A_to_B = F.grid_sample(
            certainty[None, None, ...],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, 0, 0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero(
            (D == D.min(dim=-1, keepdim=True).values)
            * (D == D.min(dim=-2, keepdim=True).values)
            * (cert_A_to_B[:, None] > self.sample_thresh),
            as_tuple=True,
        )



        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B), dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]), dim=-1)

    @torch.inference_mode()
    
    # --- 为 ONNX 准备的静态核心推理方法 ---
    def forward_core_for_onnx(self, image_a, image_b):
        """
        这是我们真正要导出到 ONNX 的核心神经网络。
        输入: 两个预处理过的图像。
        输出: 三个原始的 logits 张量。
        """
        X = torch.cat((image_a, image_b), dim=0)
        feature_pyramid_tuple = self.encoder.forward(X)
        
        f_q_pyramid_tuple = feature_pyramid_tuple
        f_s_pyramid_tuple = tuple(
            [torch.cat((f.chunk(2)[1], f.chunk(2)[0]), dim=0) for f in feature_pyramid_tuple]
        )
        
        decoder_inputs = (*f_q_pyramid_tuple, *f_s_pyramid_tuple)
        
        # decoder.forward 返回 (flow1, certainty1, certainty16)
        return self.decoder.forward(*decoder_inputs)
    
    def post_process(self, final_flow, final_certainty, low_res_certainty, im_A, im_B):
        
        #接收神经网络的原始输出和预处理后的图像，执行所有后处理步骤。
        
        hs, ws = self.h_resized, self.w_resized

        if self.attenuate_cert:
            low_res_certainty_interp = F.interpolate(low_res_certainty, size=(hs, ws), align_corners=False, mode="bilinear")
            cert_clamp_mask = (low_res_certainty_interp < 0).float()
            attenuation = 0.5 * low_res_certainty_interp * cert_clamp_mask
            certainty = final_certainty - attenuation
        else:
            certainty = final_certainty
        
        im_A_to_im_B = final_flow # Shape: [2, 2, H, W]
        certainty = certainty.sigmoid() # Shape: [2, 1, H, W]

        # 创建坐标网格 (NHWC 格式)
        im_A_coords_grid = torch.meshgrid(
            torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=im_A_to_im_B.device),
            torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=im_A_to_im_B.device),
            indexing="ij"
        )
        im_A_coords = torch.stack((im_A_coords_grid[1], im_A_coords_grid[0]), dim=-1)
        # --- 修正点 1: 使用传入的 im_A.shape[0] ---
        im_A_coords = im_A_coords.unsqueeze(0).expand(im_A.shape[0], -1, -1, -1) 

        # 边框外 flow 置信度为 0
        wrong_mask = (im_A_to_im_B.abs() > 1).sum(dim=1, keepdim=True) > 0 # Shape [2, 1, H, W]
        certainty = torch.where(wrong_mask, torch.tensor(0.0, device=certainty.device), certainty)

        # 黑色像素区域置信度为 0
        # --- 修正点 2: 直接使用传入的 im_A 和 im_B ---
        black_mask1 = (im_A < 0.03125).all(dim=1, keepdim=True)
        black_mask2 = (im_B < 0.03125).all(dim=1, keepdim=True)
        
        cert_a, cert_b = certainty.chunk(2, dim=0)
        cert_a = torch.where(black_mask1, torch.tensor(0.0, device=cert_a.device), cert_a)
        cert_b = torch.where(black_mask2, torch.tensor(0.0, device=cert_b.device), cert_b)
        certainty = torch.cat((cert_a, cert_b), dim=0)

        im_A_to_im_B_clamped = torch.clamp(final_flow, -1, 1)
        im_A_to_im_B_permuted = im_A_to_im_B_clamped.permute(0, 2, 3, 1)
        A_to_B, B_to_A = im_A_to_im_B_permuted.chunk(2, dim=0)

        q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
        s_warp = torch.cat((B_to_A, im_A_coords), dim=-1)
        
        warp = torch.cat((q_warp, s_warp), dim=2)
        
        certainty_final = torch.cat(certainty.chunk(2, dim=0), dim=3)
       
        return (
                warp[0],            
                certainty_final[0, 0],
        )
   
    
    

    
    def match(
            self,
            im_A_path,
            im_B_path,
            *args,
            batched=False,
    ):
         # 1. 预处理
        im_A = F.interpolate(im_A_path, size=(self.h_resized, self.w_resized), mode='bilinear', align_corners=False)
        im_B = F.interpolate(im_B_path, size=(self.h_resized, self.w_resized), mode='bilinear', align_corners=False)
        
        # 2. 核心神经网络推理
        final_flow, final_certainty, low_res_certainty = self.forward_core_for_onnx(im_A, im_B)
        
        # 3. 后处理
        return self.post_process(final_flow, final_certainty, low_res_certainty, im_A, im_B)
    
    


    def inference(self, batch):
        num = 5000
        b = batch['image0'].size(0)
        dense_matches, dense_certainty = self.match(batch['image0'], batch['image1'])
        sparse_matches = self.sample(dense_matches, dense_certainty, num)[0]

        batch.update({
            'hw0_i': batch['image0'].shape[2:],
            'hw1_i': batch['image1'].shape[2:]
        })

        h1, w1 = batch['hw0_i']
        kpts1 = sparse_matches[:, :, :2]
        kpts1 = torch.stack((w1 * (kpts1[:, :, 0] + 1) / 2,
                             h1 * (kpts1[:, :, 1] + 1) / 2,), dim=-1,)
        kpts1 *= batch['scale0'].unsqueeze(1)

        h2, w2 = batch['hw1_i']
        kpts2 = sparse_matches[:, :, 2:]
        kpts2 = torch.stack((w2 * (kpts2[:, :, 0] + 1) / 2,
                             h2 * (kpts2[:, :, 1] + 1) / 2,), dim=-1,)
        kpts2 *= batch['scale1'].unsqueeze(1)

        # b_ids = torch.zeros_like(kpts1[:, 0], device=kpts1.device).long()
        b_ids = torch.arange(b).unsqueeze(1).repeat(1, num).to(kpts1.device).reshape(-1)

        batch.update({
            'm_bids': b_ids,
            "mkpts0_f": kpts1.reshape(-1, 2),
            "mkpts1_f": kpts2.reshape(-1, 2),
        })


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        blocks,
        hidden_dim,
        out_dim,
        is_classifier=False,
        pos_enc=True,
        learned_embeddings=False,
        embedding_dim=None,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._scales = [16]
        self.is_classifier = is_classifier
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings
        

    def scales(self):
        return self._scales.copy()

    def forward(self, gp_posterior, features):
        def get_grid(b, h, w, device):
            grid = torch.meshgrid(
                *[
                    torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
                    for n in (b, h, w)
                ]
            )
            grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
            return grid

        B, C, H, W = gp_posterior.shape
        x = torch.cat((gp_posterior, features), dim=1)
        B, C, H, W = x.shape
        grid = get_grid(B, H, W, x.device).reshape(B, H * W, 2)
      
        pos_enc = 0
        tokens = x.reshape(B, C, H * W).permute(0, 2, 1) + pos_enc
        z = self.blocks(tokens)
        out = self.to_out(z)
        out = out.permute(0, 2, 1).reshape(B, self.out_dim, H, W)
        warp, certainty = out[:, :-1], out[:, -1:]
        return warp, certainty


def kde(x, std=0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    x = x.half()  # Do it in half precision TODO: remove hardcoding
    scores = (-torch.cdist(x, x) ** 2 / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density



def local_correlation(feature0, feature1, local_radius, padding_mode="zeros", flow=None, sample_mode="bilinear"):
    """
    A vectorized version of local_correlation, suitable for ONNX export
    with dynamic batch size.
    """
    r = local_radius
    K = (2 * r + 1) ** 2
    B, c, h, w = feature0.shape

    coords = flow.permute(0, 2, 3, 1)  # Shape: [B, h, w, 2]
        
    dtype = feature0.dtype
    # Create the local window offset grid
    local_window_h = torch.linspace(-2*r/h, 2*r/h, 2*r+1, device=feature0.device,dtype=dtype)
    local_window_w = torch.linspace(-2*r/w, 2*r/w, 2*r+1, device=feature0.device,dtype=dtype)
    grid_y, grid_x = torch.meshgrid(local_window_h, local_window_w, indexing='ij')
    local_window = torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, 1, K, 2) # Shape: [1, 1, 1, K, 2]

    # --- Vectorized Sampling ---
    # Expand coords and local_window to perform broadcasted addition
    # coords shape:         [B, h, w, 1, 2]
    # local_window shape:   [1, 1, 1, K, 2]
    # Resulting shape:      [B, h, w, K, 2]
    local_window_coords = coords.unsqueeze(-2) + local_window
        
    # Reshape for grid_sample: grid_sample expects grid of shape [N, H_out, W_out, 2]
    # We can treat the K neighbors as a single long width dimension.
    # Shape becomes: [B, h, w * K, 2]
    local_window_coords = local_window_coords.reshape(B, h, w * K, 2)
        
    # Sample all neighbors for all batch items at once
    window_feature = F.grid_sample(
        feature1,
        local_window_coords.to(torch.float32), 
        padding_mode=padding_mode, 
        align_corners=False, 
        mode=sample_mode
    )
        
    # Reshape the sampled features back to have the neighbor dimension
    # Shape: [B, c, h, w * K] -> [B, c, h, w, K]
    window_feature = window_feature.reshape(B, c, h, w, K)
        
    # --- Vectorized Correlation Calculation ---
    # feature0 shape:           [B, c, h, w] -> [B, c, h, w, 1]
    # window_feature shape:     [B, c, h, w, K]
    # Broadcasting will make feature0 match the shape of window_feature
        
    # Perform element-wise multiplication and sum over the channel dimension
    corr = (feature0.unsqueeze(-1) / (c**0.5) * window_feature).sum(dim=1)
        
    # Final shape of corr is [B, h, w, K]. Permute to [B, K, h, w].
    return corr.permute(0, 3, 1, 2)


@torch.no_grad()
def cls_to_flow_refine(cls):
    B, C, H, W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / res, 1 - 1 / res, steps=res, device=device)
            for _ in range(2)
        ]
    )
    G = torch.stack([G[1], G[0]], dim=-1).reshape(C, 2)
    cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices

    index = (
        torch.stack((mode - 1, mode, mode + 1, mode - res, mode + res), dim=1)
        .clamp(0, C - 1)
        .long()
    )
    neighbours = torch.gather(cls, dim=1, index=index)[..., None]
    flow = (
        neighbours[:, 0] * G[index[:, 0]]
        + neighbours[:, 1] * G[index[:, 1]]
        + neighbours[:, 2] * G[index[:, 2]]
        + neighbours[:, 3] * G[index[:, 3]]
        + neighbours[:, 4] * G[index[:, 4]]
    )
    tot_prob = neighbours.sum(dim=1)
    flow = flow / tot_prob
    return flow


def RoMa(img_size):
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(
            *[Block(decoder_dim, 8, attn_class=Attention) for _ in range(5)]
        ),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        pos_enc=False,
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

  
    conv_refiner = nn.ModuleDict({
    "16": ConvRefinerWithCorr(
        in_dim=2 * 512 + 128 + (2 * 7 + 1) ** 2,
        hidden_dim=2 * 512 + 128 + (2 * 7 + 1) ** 2,
        out_dim=2 + 1,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb_dim=128,
        local_corr_radius=7,
        # Pass the original unused parameters, which will be caught by **kwargs
        corr_in_other=True, 
        disable_local_corr_grad=disable_local_corr_grad, 
        bn_momentum=0.01,
    ),
    "8": ConvRefinerWithCorr(
        in_dim=2 * 512 + 64 + (2 * 3 + 1) ** 2,
        hidden_dim=2 * 512 + 64 + (2 * 3 + 1) ** 2,
        out_dim=2 + 1,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb_dim=64,
        local_corr_radius=3,
        corr_in_other=True,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
    ),
    "4": ConvRefinerWithCorr(
        in_dim=2 * 256 + 32 + (2 * 2 + 1) ** 2,
        hidden_dim=2 * 256 + 32 + (2 * 2 + 1) ** 2,
        out_dim=2 + 1,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb_dim=32,
        local_corr_radius=2,
        corr_in_other=True,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
    ),
    "2": ConvRefinerNoCorr(
        in_dim=2 * 64 + 16,
        hidden_dim=128 + 16,
        out_dim=2 + 1,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        # Pass the original unused parameters
        displacement_emb=displacement_emb,
        displacement_emb_dim=16,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
    ),
    "1": ConvRefinerNoCorr(
        in_dim=2 * 9 + 6,
        hidden_dim=24,
        out_dim=2 + 1,
        kernel_size=kernel_size,
        dw=dw,
        hidden_blocks=hidden_blocks,
        displacement_emb=displacement_emb,
        displacement_emb_dim=6,
        disable_local_corr_grad=disable_local_corr_grad,
        bn_momentum=0.01,
    ),
    })

    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    
    gp16 = GP16(
        kernel=CosKernel,
        gp_dim=gp_dim,
    )
    
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict(
        {
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
        }
    )
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )
    assert img_size is not None
    assert isinstance(img_size, list)
    assert len(img_size) <= 2
    if len(img_size) == 1: img_size = img_size * 2
    h, w = img_size
    encoder = CNNandDinov2(
    )
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w)
    return matcher
