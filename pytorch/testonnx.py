import cv2
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
import time
import math

# ---------------- 工具函数 (从原代码复刻) ----------------

def kde(x, std=0.1):
    # 使用高斯核估计密度
    # x: [N, 4] (matches)
    # 在 GPU 上计算 cdist 速度快很多
    x = x.half() # 使用半精度加速
    scores = (-torch.cdist(x, x) ** 2 / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density

def get_padding_size(image, h, w):
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

# ---------------- ONNX 推理类 ----------------

class Matcher_roma_onnx:
    def __init__(self, onnx_path='onnx/roma_core.onnx', img_size=504, device=None):
        '''
        初始化 RoMa ONNX 模型
        '''
        # 1. 配置 Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"🚀 Initializing Matcher with Target Device: {self.device.upper()}")

        # 2. 配置 ONNX Runtime Session
        providers = []
        if self.device == 'cuda':
            # 检查 onnxruntime-gpu 是否真正可用
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                print("⚠️ Warning: 'CUDAExecutionProvider' not found. Fallback to CPU. Please install onnxruntime-gpu.")
                self.device = 'cpu'
                providers = ['CPUExecutionProvider']
            else:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ]
        else:
            providers = ['CPUExecutionProvider']

        print(f"✅ Active ONNX Providers: {[p if isinstance(p, str) else p[0] for p in providers]}")

        # 加载模型
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            raise e
            
        self.img_size = img_size
        
        # 预计算常量（减少推理时的重复计算）
        self.sample_thresh = 0.05
        self.attenuate_cert = True

    @staticmethod
    def _resize_image(image: np.ndarray, size):
        h, w = image.shape[:2]
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA if max(h, w)>max(size) else cv2.INTER_LINEAR)

    def _preprocess(self, image: np.ndarray):
        # 与原 PyTorch 逻辑保持一致
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            image = image[..., :3]
        if not ((image.max() <= 1.0 and image.dtype == np.float32) or (image.max() <= 255)):
            raise ValueError("Input image type error")
        if image.max() > 1.5:
            image = image.astype(np.float32) / 255.
        if np.mean(image[..., 0]) > np.mean(image[..., 2]):  # BGR -> RGB
            image = image[..., ::-1]
        
        h, w = image.shape[:2]
        scale = 1.
        if max(h, w) > self.img_size:
            scale = self.img_size / max(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            image = self._resize_image(image, (nw, nh))
            
        # Transpose to CHW
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        return image_t, scale

    def _post_process_flow(self, final_flow, final_certainty, low_res_certainty, im_A_shape, im_B_shape):
        """
        完全复刻 RegressionMatcher.post_process，但确保在 GPU 上运行以加速
        """
        hs, ws = self.img_size, self.img_size

        # 确保输入都在 GPU 上 (ONNX 输出是 numpy，这里转回 Tensor 并上 GPU)
        # 如果已经在 GPU 上则不操作
        if not isinstance(final_flow, torch.Tensor):
            final_flow = torch.from_numpy(final_flow).to(self.device)
            final_certainty = torch.from_numpy(final_certainty).to(self.device)
            low_res_certainty = torch.from_numpy(low_res_certainty).to(self.device)

        # 1. 处理 Certainty Attenuation
        if self.attenuate_cert:
            # 这里的 interpolate 必须用 scale_factor=14，与导出时的逻辑对应
            low_res_certainty_interp = F.interpolate(
                low_res_certainty, 
                scale_factor=14, 
                align_corners=False, 
                mode="bilinear"
            )
            cert_clamp_mask = (low_res_certainty_interp < 0).float()
            attenuation = 0.5 * low_res_certainty_interp * cert_clamp_mask
            certainty = final_certainty - attenuation
        else:
            certainty = final_certainty
        
        certainty = certainty.sigmoid()

        # 2. 创建坐标网格 (在 GPU 上非常快)
        # 注意：这里 im_A_shape[0] 是 batch_size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=self.device),
            torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=self.device),
            indexing="ij"
        )
        im_A_coords = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(im_A_shape[0], -1, -1, -1) # [B, H, W, 2]

        # 3. 处理 Flow
        im_A_to_im_B = final_flow # [B, 2, H, W]
        
        # 边框外置信度置 0
        wrong_mask = (im_A_to_im_B.abs() > 1).sum(dim=1, keepdim=True) > 0
        certainty = torch.where(wrong_mask, torch.tensor(0.0, device=self.device), certainty)

        # (省略了 Black Mask 检查以加速，如果图片有纯黑边框需求可加回，但需要在 GPU 上做)
        
        im_A_to_im_B_clamped = torch.clamp(im_A_to_im_B, -1, 1)
        im_A_to_im_B_permuted = im_A_to_im_B_clamped.permute(0, 2, 3, 1) # [B, H, W, 2]
        
        # 切分双向 Flow
        A_to_B, B_to_A = im_A_to_im_B_permuted.chunk(2, dim=0)
        
        # 构造 Warp
        q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
        s_warp = torch.cat((B_to_A, im_A_coords), dim=-1)
        warp = torch.cat((q_warp, s_warp), dim=2) # [1, H, 2W, 4]
        
        certainty_final = torch.cat(certainty.chunk(2, dim=0), dim=3) # [1, 1, H, 2W]

        return warp[0], certainty_final[0, 0]

    def _sample(self, dense_matches, dense_certainty, num=5000):
        """
        在 GPU 上执行采样和 KDE，这是解决这一步慢的关键
        """
        upper_thresh = self.sample_thresh
        dense_certainty = dense_certainty.clone()
        
        # 过滤低置信度
        dense_certainty[dense_certainty > upper_thresh] = 1
        
        matches = dense_matches.reshape(-1, 4)
        certainty = dense_certainty.reshape(-1)

        if certainty.sum() < 1e-6:
            certainty = certainty + 1e-8

        # 1. 第一次采样 (Weighted Random)
        # torch.multinomial 在 CUDA 上也是优化的
        expansion_factor = 4 
        num_samples = min(expansion_factor * num, len(certainty))
        
        good_samples = torch.multinomial(certainty, num_samples=num_samples, replacement=False)
        good_matches = matches[good_samples]
        good_certainty = certainty[good_samples] # 这里简化了，原代码用了 backup certainty，这里直接用

        # 2. KDE 密度估计 (耗时大户，必须在 GPU)
        # kde 函数内部会转 half 精度加速
        density = kde(good_matches, std=0.1)
        
        p = 1 / (density + 1)
        p[density < 10] = 1e-7 
        
        # 3. 第二次采样 (Balanced)
        balanced_samples_idx = torch.multinomial(p, num_samples=min(num, len(good_certainty)), replacement=False)
        
        return good_matches[balanced_samples_idx], good_certainty[balanced_samples_idx]

    def extract_matches(self, img_A, img_B):
        # 1. CPU 预处理
        tt0 = time.time()
        image0_t, scale0 = self._preprocess(img_A)
        image1_t, scale1 = self._preprocess(img_B)
        
        # 计算 Padding
        pad = lambda im: get_padding_size(im, self.img_size, self.img_size)
        ow0, oh0, pl0, pr0, pt0, pb0 = pad(image0_t)
        ow1, oh1, pl1, pr1, pt1, pb1 = pad(image1_t)
        
        image0_pad = F.pad(image0_t, (pl0, pr0, pt0, pb0))
        image1_pad = F.pad(image1_t, (pl1, pr1, pt1, pb1))
        tt1 = time.time()
        print(f"1 Time: {tt1 - tt0:.4f}s")
        
        # 2. ONNX 推理
        # 输入转 Numpy
        ort_inputs = {
            self.session.get_inputs()[0].name: image0_pad.numpy(),
            self.session.get_inputs()[1].name: image1_pad.numpy()
        }
        
        # Run!
        # 注意：sess.run 返回的是 list of numpy arrays (在 CPU 上)
        # 这是 ONNX Runtime Python API 的限制，除非使用 IO Binding (代码极其复杂)
        # 但我们只要在这里拿到结果后立刻扔回 GPU 即可
        outs = self.session.run(None, ort_inputs)
        final_flow, final_certainty, low_res_certainty = outs
        tt2 = time.time()
        print(f"2 Time: {tt2 - tt1:.4f}s")
        
        # 3. 后处理 (Post Process) - 转回 GPU 加速
        dense_matches, dense_certainty = self._post_process_flow(
            final_flow, final_certainty, low_res_certainty, 
            image0_pad.shape, image1_pad.shape
        )
        
        # 4. 采样 (Sampling) - 在 GPU 上运行
        sparse_matches, _ = self._sample(dense_matches, dense_certainty, num=5000)
        
        # 5. 坐标还原 (GPU 上计算)
        h0, w0 = self.img_size, self.img_size
        h1, w1 = self.img_size, self.img_size # padding 后是一样的
        
        kpts0 = sparse_matches[:, :2]
        kpts0 = torch.stack((w0 * (kpts0[:, 0] + 1) / 2, h0 * (kpts0[:, 1] + 1) / 2), dim=-1)
        kpts1 = sparse_matches[:, 2:]
        kpts1 = torch.stack((w1 * (kpts1[:, 0] + 1) / 2, h1 * (kpts1[:, 1] + 1) / 2), dim=-1)
        
        # 去掉 Padding
        kpts0 -= torch.tensor([pl0, pt0], device=self.device).unsqueeze(0)
        kpts1 -= torch.tensor([pl1, pt1], device=self.device).unsqueeze(0)
        
        # 过滤出界点
        mask = (
            (kpts0[:, 0] > 0) & (kpts0[:, 0] < (ow0 - 1)) &
            (kpts1[:, 0] > 0) & (kpts1[:, 0] < (ow1 - 1)) &
            (kpts0[:, 1] > 0) & (kpts0[:, 1] < (oh0 - 1)) &
            (kpts1[:, 1] > 0) & (kpts1[:, 1] < (oh1 - 1))
        )
        
        # 最终转回 CPU numpy 供 cv2 使用
        kpts0_np = (kpts0[mask] / scale0).cpu().numpy()
        kpts1_np = (kpts1[mask] / scale1).cpu().numpy()
        tt3 = time.time()
        print(f"3 Time: {tt3 - tt2:.4f}s")
        return kpts0_np, kpts1_np

    def roma_H(self, img_A, img_B, ransac_thresh=4.0, confidence=0.99, max_iter=10000):
        print("\n--- Start RoMa ONNX Matching ---")
        t0 = time.time()
        
        # 1. 提取匹配点
        kptsA, kptsB = self.extract_matches(img_A, img_B)
        t1 = time.time()
        print(f"Feature Extraction + Sampling Time: {t1 - t0:.4f}s")
        print(f"Num Keypoints: {len(kptsA)}")

        if len(kptsA) < 4:
            print("❌ Not enough matches found.")
            return None

        # 2. 计算 Homography (使用 OpenCV CPU)
        H, mask = cv2.findHomography(
            kptsA, kptsB,
            cv2.USAC_MAGSAC, ransacReprojThreshold=ransac_thresh,
            confidence=confidence, maxIters=max_iter
        )
        
        t2 = time.time()
        print(f"Homography Time: {t2 - t1:.4f}s")
        print(f"Total Time: {t2 - t0:.4f}s")
        
        return H

# ----------- 用法例子 ----------
if __name__ == '__main__':
    # 加载图片
    # 请确保当前目录下有 a1.jpg 和 a2.jpg，或者修改为存在的路径
    # 如果没有图片，可以用 np.zeros 生成假图测试流程
    img0_path = './a1.jpg'
    img1_path = './a2.jpg'
    
    try:
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        if img0 is None or img1 is None: raise ValueError("Image not found")
    except Exception:
        print("⚠️ Test images not found, generating random images for performance test.")
        img0 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        img1 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

    # 初始化 ONNX 匹配器
    # ⚠️ 请确保 onnx/roma_core.onnx 存在
    try:
        matcher = Matcher_roma_onnx(onnx_path='./onnx/roma_core.onnx')
        
        # 预热一次 (第一次推理通常较慢)
        print("\nWarmup inference...")
        matcher.roma_H(img0, img1)
        
        # 正式运行
        print("\nRunning Inference...")
        start = time.time()
        H = matcher.roma_H(img0, img1)
        end = time.time()
        print ("spend time:",str(end-start))
        if H is not None:
            print("\nFound H matrix:\n", H)
        else:
            print("\nFailed to find H matrix.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")