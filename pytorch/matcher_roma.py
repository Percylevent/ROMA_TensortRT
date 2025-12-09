import cv2
import torch
import numpy as np
import time
import warnings
import torch.nn.functional as F
from roma import RoMa  # 假设你的 RoMa 类定义在 roma.py 中

def get_padding_size(image, h, w):
    """
    计算 Padding 参数，使图像居中
    """
    orig_width = image.shape[3]
    orig_height = image.shape[2]
    aspect_ratio = w / h
    new_width = max(orig_width, int(orig_height * aspect_ratio))
    new_height = max(orig_height, int(orig_width / aspect_ratio))
    pad_height = new_height - orig_height
    pad_width = new_width - orig_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return orig_width, orig_height, pad_left, pad_right, pad_top, pad_bottom

class Matcher_roma:
    def __init__(self, ckpt_path='weights/gim_roma_100h.ckpt', img_size=504, device=None):
        '''
        Args:
            ckpt_path: 权重路径
            img_size: 输入网络的最长边大小 (官方示例默认 672，你也可用 504)
            device: 运行设备
        '''
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # 初始化模型
        self.model = RoMa(img_size=[self.img_size])
        
        # 加载权重 (处理 model. 前缀)
        print(f"Loading weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k.replace('model.', '', 1)] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.eval().to(self.device)

    def _preprocess(self, image: np.ndarray):
        """
        官方风格预处理: 保持长宽比 resize，归一化，转 Tensor
        返回: (tensor [1,3,H,W], scale_factor)
        """
        # 1. 确保 RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., :3]
        # BGR 转 RGB
        # (简单判断：如果 B 通道均值远大于 R 通道，可能是 BGR，或者由外部保证输入为 RGB)
        # 这里假设输入即为读取的 cv2 image (BGR)，转为 RGB
        image = image[..., ::-1] 

        # 2. Resize (保持长宽比，最长边 = img_size)
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        # 如果图片比 img_size 小，通常也建议放大，或者根据需求决定
        # 这里逻辑是：统一缩放到 max_dim = img_size
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 3. 归一化 & 转 Tensor
        image_t = image_resized.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_t.transpose(2, 0, 1)).float().unsqueeze(0) # (1, 3, H, W)
        
        return image_t, scale, (h, w)

    @torch.inference_mode()
    def extract_matches(self, img_A, img_B):
        """
        核心推理函数
        Args:
            img_A, img_B: numpy array (H, W, 3) BGR format (cv2 default)
        Returns:
            kpts0, kpts1: numpy array (N, 2) 原始图像上的坐标
        """
        # 1. 预处理
        image0_t, scale0, (orig_h0, orig_w0) = self._preprocess(img_A)
        image1_t, scale1, (orig_h1, orig_w1) = self._preprocess(img_B)
        
        image0_t = image0_t.to(self.device)
        image1_t = image1_t.to(self.device)

        # 2. 计算 Padding (使图像变为 img_size x img_size 的正方形)
        # 注意：这里传入的宽高都是 self.img_size
        _, _, pl0, pr0, pt0, pb0 = get_padding_size(image0_t, self.img_size, self.img_size)
        _, _, pl1, pr1, pt1, pb1 = get_padding_size(image1_t, self.img_size, self.img_size)

        # 3. Apply Padding
        image0_pad = F.pad(image0_t, (pl0, pr0, pt0, pb0))
        image1_pad = F.pad(image1_t, (pl1, pr1, pt1, pb1))

        # 获取 Padding 后的实际尺寸 (通常等于 img_size)
        h0_pad, w0_pad = image0_pad.shape[-2:]
        h1_pad, w1_pad = image1_pad.shape[-2:]

        # 4. 模型推理
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dense_matches, dense_certainty = self.model.match(image0_pad, image1_pad)
            sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, 5000)

        # 5. 坐标还原 (完全遵循官方逻辑)
        # sparse_matches 格式为 [-1, 1] 的归一化坐标
        kpts0_norm = sparse_matches[:, :2]
        kpts1_norm = sparse_matches[:, 2:]

        # A. 反归一化：映射回 Padding 后的像素坐标 [0, img_size]
        kpts0 = torch.stack((
            w0_pad * (kpts0_norm[:, 0] + 1) / 2, 
            h0_pad * (kpts0_norm[:, 1] + 1) / 2
        ), dim=-1)
        
        kpts1 = torch.stack((
            w1_pad * (kpts1_norm[:, 0] + 1) / 2, 
            h1_pad * (kpts1_norm[:, 1] + 1) / 2
        ), dim=-1)

        # B. 去除 Padding 偏移
        kpts0 -= torch.tensor([[pl0, pt0]], device=self.device)
        kpts1 -= torch.tensor([[pl1, pt1]], device=self.device)

        # C. 边界过滤 (Masking) - 官方代码中的关键步骤
        # 过滤掉落在 Padding 区域或者超出原始 Resize 图像边界的点
        # 注意：此时 kpts 还是相对于 Resize 后的图像，尚未除以 scale
        # Resize 后的宽为 (orig_w * scale)
        valid_w0, valid_h0 = int(round(orig_w0 * scale0)), int(round(orig_h0 * scale0))
        valid_w1, valid_h1 = int(round(orig_w1 * scale1)), int(round(orig_h1 * scale1))

        mask = (
            (kpts0[:, 0] > 0) & (kpts0[:, 0] <= (valid_w0 - 1)) &
            (kpts0[:, 1] > 0) & (kpts0[:, 1] <= (valid_h0 - 1)) &
            (kpts1[:, 0] > 0) & (kpts1[:, 0] <= (valid_w1 - 1)) &
            (kpts1[:, 1] > 0) & (kpts1[:, 1] <= (valid_h1 - 1))
        )
        
        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]
        
        # D. 反缩放：映射回原始图像分辨率
        kpts0 /= scale0
        kpts1 /= scale1

        # 转为 numpy
        return kpts0.cpu().numpy(), kpts1.cpu().numpy()

    def roma_H(self, img_A, img_B, ransac_thresh=4.0):
        """
        计算单应性矩阵
        """
        print("Starting matching...")
        t0 = time.time()
        
        kptsA, kptsB = self.extract_matches(img_A, img_B)
        
        if len(kptsA) < 8: # 单应性至少需要4对，稳健起见设8
            print("Not enough matches found.")
            return None

        # 使用官方示例推荐的 USAC_MAGSAC
        H, mask = cv2.findHomography(
            kptsA, kptsB,
            cv2.USAC_MAGSAC, 
            ransacReprojThreshold=ransac_thresh,
            confidence=0.99999, 
            maxIters=10000
        )
        
        t1 = time.time()
        print(f"Match done. Found {len(kptsA)} matches. Time: {t1 - t0:.4f}s")
        
        return H

# ----------- 测试部分 ----------
if __name__ == '__main__':
    # 路径配置
    img0_path = './a1.jpg'
    img1_path = './a2.jpg'
    ckpt_path = './gim_roma_100h.ckpt' # 请确保路径正确

    # 读取图片
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
    if img0 is None or img1 is None:
        print("Error reading images.")
        exit()

    # 初始化匹配器 (官方示例用 672，你也可以用 504)
    matcher = Matcher_roma(ckpt_path, img_size=504) 
    
    # 计算 H
    H = matcher.roma_H(img0, img1)
    print("Found Homography:\n", H)

    # 简单的可视化验证
    if H is not None:
        h, w = img0.shape[:2]
        # 将图0 warp 到图1
        warped_img0 = cv2.warpPerspective(img0, H, (img1.shape[1], img1.shape[0]))
        # 叠加显示
        blend = cv2.addWeighted(img1, 0.5, warped_img0, 0.5, 0)
        cv2.imwrite("roma_verify_warp.jpg", blend)
        print("Saved verification image to roma_verify_warp.jpg")