import tensorrt as trt
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import os

# ---------------- 辅助工具 ----------------

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

def kde(x, std=0.1):
    x = x.half() 
    scores = (-torch.cdist(x, x) ** 2 / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density

# ---------------- TensorRT 推理类 ----------------

class Matcher_roma_trt:
    def __init__(self, engine_path, img_size=504):
        self.img_size = img_size
        self.device = torch.device('cuda')
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"🚀 Loading TensorRT Engine from {engine_path}...")
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"❌ Error loading engine: {e}")
            raise e
            
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()
        
        # 分配显存
        self.inputs = []
        self.outputs = []
        self.tensor_name_to_ptr = {} 
        self.buffers = [] 
        
        num_io_tensors = self.engine.num_io_tensors
        for i in range(num_io_tensors):
            name = self.engine.get_tensor_name(i)
            raw_shape = self.engine.get_tensor_shape(name)
            dims = [1 if d == -1 else d for d in raw_shape]
            shape = tuple(dims)
            
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = self._trt_dtype_to_torch(trt_dtype)
            
            tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
            self.buffers.append(tensor)
            ptr = tensor.data_ptr()
            self.tensor_name_to_ptr[name] = ptr
            
            mode = self.engine.get_tensor_mode(name)
            binding = {'index': i, 'name': name, 'tensor': tensor, 'ptr': ptr}
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        
        for inp in self.inputs:
            self.context.set_input_shape(inp['name'], inp['tensor'].shape)

        self.sample_thresh = 0.05
        self.attenuate_cert = True
        self.stream = torch.cuda.Stream()
        
        print("✅ TensorRT Engine Initialized!")

    def _trt_dtype_to_torch(self, trt_dtype):
        if trt_dtype == trt.float32: return torch.float32
        elif trt_dtype == trt.float16: return torch.float16
        elif trt_dtype == trt.int32: return torch.int32
        elif trt_dtype == trt.int8: return torch.int8
        elif trt_dtype == trt.bool: return torch.bool
        else: raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")

    def _preprocess_official(self, image: np.ndarray):
        """
        严格遵循官方示例的预处理逻辑：保持长宽比缩放
        """
        # 1. 保证 RGB
        if image.ndim == 2: image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4: image = image[..., :3]
        # BGR to RGB (假设输入是 cv2 读取的)
        image = image[..., ::-1] 

        # 2. Resize (保持长宽比，最长边 = img_size)
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 3. Normalize & Tensor
        image_t = image_resized.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_t.transpose(2, 0, 1)).float().unsqueeze(0)
        
        # 返回 resize 后的 tensor, 缩放比例, 以及 resize 后的实际尺寸(用于后续 mask)
        return image_t, scale, (new_h, new_w)

    def _post_process_flow(self, final_flow, final_certainty, low_res_certainty, im_A, im_B):
        """
        密集流后处理 + 黑边置信度抑制
        """
        hs, ws = self.img_size, self.img_size
        
        if self.attenuate_cert:
            low_res_certainty_interp = F.interpolate(
                low_res_certainty, size=(hs, ws), align_corners=False, mode="bilinear"
            )
            cert_clamp_mask = (low_res_certainty_interp < 0).float()
            attenuation = 0.5 * low_res_certainty_interp * cert_clamp_mask
            certainty = final_certainty - attenuation
        else:
            certainty = final_certainty
        
        certainty = certainty.sigmoid()

        # --- TensorRT 特有优化：强制抑制黑边区域的置信度 ---
        # 这一步是为了防止采样器选中 Padding 区域的噪点
        black_mask1 = (im_A < 0.03125).all(dim=1, keepdim=True)
        black_mask2 = (im_B < 0.03125).all(dim=1, keepdim=True)
        cert_a, cert_b = certainty.chunk(2, dim=0)
        cert_a = torch.where(black_mask1, torch.tensor(0.0, device=self.device), cert_a)
        cert_b = torch.where(black_mask2, torch.tensor(0.0, device=self.device), cert_b)
        certainty = torch.cat((cert_a, cert_b), dim=0)
        # -----------------------------------------------

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=self.device),
            torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=self.device),
            indexing="ij"
        )
        # 修复维度: 使用 im_A.shape[0] (即1)
        im_coords = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(im_A.shape[0], -1, -1, -1)

        wrong_mask = (final_flow.abs() > 1).sum(dim=1, keepdim=True) > 0
        certainty = torch.where(wrong_mask, torch.tensor(0.0, device=self.device), certainty)

        im_A_to_im_B_clamped = torch.clamp(final_flow, -1, 1)
        im_A_to_im_B_permuted = im_A_to_im_B_clamped.permute(0, 2, 3, 1)
        A_to_B, B_to_A = im_A_to_im_B_permuted.chunk(2, dim=0)
        
        warp = torch.cat((
            torch.cat((im_coords, A_to_B), dim=-1),
            torch.cat((B_to_A, im_coords), dim=-1)
        ), dim=2)
        
        certainty_final = torch.cat(certainty.chunk(2, dim=0), dim=3)
        return warp[0], certainty_final[0, 0]

    def _sample(self, dense_matches, dense_certainty, num=5000):
        # 固定随机种子
        #torch.manual_seed(0)
        #if torch.cuda.is_available(): torch.cuda.manual_seed(0)
            
        dense_certainty[dense_certainty > self.sample_thresh] = 1
        matches = dense_matches.reshape(-1, 4)
        certainty = dense_certainty.reshape(-1)
        if certainty.sum() < 1e-6: certainty = certainty + 1e-8

        expansion_factor = 4
        num_samples = min(expansion_factor * num, len(certainty))
        
        good_samples = torch.multinomial(certainty, num_samples=num_samples, replacement=False)
        good_matches = matches[good_samples]
        
        density = kde(good_matches, std=0.1)
        p = 1 / (density + 1)
        p[density < 10] = 1e-7 
        
        balanced_samples_idx = torch.multinomial(p, num_samples=min(num, len(good_samples)), replacement=False)
        return good_matches[balanced_samples_idx]

    def extract_matches(self, img_A, img_B):
        # 1. 官方逻辑预处理
        image0_t, scale0, (vh0, vw0) = self._preprocess_official(img_A)
        image1_t, scale1, (vh1, vw1) = self._preprocess_official(img_B)
        
        # 2. 计算并应用 Padding (变成 504x504)
        pad = lambda im: get_padding_size(im, self.img_size, self.img_size)
        ow0, oh0, pl0, pr0, pt0, pb0 = pad(image0_t)
        ow1, oh1, pl1, pr1, pt1, pb1 = pad(image1_t)
        
        image0_pad = F.pad(image0_t, (pl0, pr0, pt0, pb0)).contiguous()
        image1_pad = F.pad(image1_t, (pl1, pr1, pt1, pb1)).contiguous()
        
        h_pad, w_pad = image0_pad.shape[-2:] # 504, 504
        
        # 3. TensorRT 推理
        with torch.cuda.stream(self.stream):
            self.inputs[0]['tensor'].copy_(image0_pad, non_blocking=True)
            self.inputs[1]['tensor'].copy_(image1_pad, non_blocking=True)
            for name, ptr in self.tensor_name_to_ptr.items():
                self.context.set_tensor_address(name, ptr)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        
        output_map = {x['name']: x['tensor'] for x in self.outputs}
        
        # 4. 传回 GPU 进行后处理计算
        im_A_gpu = image0_pad.to(self.device, non_blocking=True)
        im_B_gpu = image1_pad.to(self.device, non_blocking=True)

        dense_matches, dense_certainty = self._post_process_flow(
            output_map['final_flow'], output_map['final_certainty'], output_map['low_res_certainty'], 
            im_A_gpu, im_B_gpu
        )
        
        sparse_matches = self._sample(dense_matches, dense_certainty, num=5000)
        
        # 5. 坐标还原 (严格遵循官方逻辑)
        # 归一化坐标 [-1, 1]
        kpts0_norm = sparse_matches[:, :2]
        kpts1_norm = sparse_matches[:, 2:]
        
        # A. 映射回 Padding 后的像素坐标 [0, 504]
        kpts0 = torch.stack((w_pad * (kpts0_norm[:, 0] + 1) / 2, h_pad * (kpts0_norm[:, 1] + 1) / 2), dim=-1)
        kpts1 = torch.stack((w_pad * (kpts1_norm[:, 0] + 1) / 2, h_pad * (kpts1_norm[:, 1] + 1) / 2), dim=-1)
        
        # B. 减去 Padding 偏移
        kpts0 -= torch.tensor([[pl0, pt0]], device=self.device)
        kpts1 -= torch.tensor([[pl1, pt1]], device=self.device)
        
        # C. 官方 Mask 过滤 (剔除 Padding 区域和越界点)
        mask = (
            (kpts0[:, 0] > 0) & (kpts0[:, 0] <= (vw0 - 1)) &
            (kpts0[:, 1] > 0) & (kpts0[:, 1] <= (vh0 - 1)) &
            (kpts1[:, 0] > 0) & (kpts1[:, 0] <= (vw1 - 1)) &
            (kpts1[:, 1] > 0) & (kpts1[:, 1] <= (vh1 - 1))
        )
        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]
        
        # D. 除以缩放比例 (还原回原图尺寸)
        kpts0 /= scale0
        kpts1 /= scale1
        
        return kpts0.cpu().numpy(), kpts1.cpu().numpy()

    def roma_H(self, img_A, img_B):
        try:
            kptsA, kptsB = self.extract_matches(img_A, img_B)
            
            if len(kptsA) < 8: return None
            
            # 使用官方推荐的 USAC_MAGSAC
            H, mask = cv2.findHomography(
                kptsA, kptsB,
                cv2.USAC_MAGSAC, ransacReprojThreshold=4.0,
                confidence=0.99999, maxIters=10000
            )
            return H
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None

def wrap_and_save(img0, img1, H, save_path):
    """
    可视化 Warp 结果并保存
    """
    if H is None:
        print("H is None, cannot warp.")
        return

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    # 将图0 变换到 图1 的视角
    # 输出画布大小应当是图1的大小
    warped_img0 = cv2.warpPerspective(img0, H, (w1, h1))
    
    # 简单的加权叠加
    alpha = 0.5
    blend = cv2.addWeighted(img1, alpha, warped_img0, 1 - alpha, 0)
    
    cv2.imwrite(save_path, blend)
    print(f"✅ Warp visualization saved to {save_path}")

# ---------------- 主程序 ----------------

if __name__ == '__main__':
    ENGINE_PATH = './roma_core_ori_fp16.engine' 
    img0_path = './a1.jpg'
    img1_path = './a2.jpg'
    
    if not os.path.exists(ENGINE_PATH):
        print(f"❌ Error: Engine file not found at {ENGINE_PATH}")
        exit()
        
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
    matcher = Matcher_roma_trt(engine_path=ENGINE_PATH)
    for i in range(100):
        # 1. 计算 H
        print("\nCalculating Homography...")
        start_time = time.time()
        H = matcher.roma_H(img0, img1)
        end_time = time.time()
    
        print(f"Time taken: {end_time - start_time:.4f}s")
        print("Found H:\n", H)
    
        # 2. 保存 Warp 效果图
        if H is not None:
            wrap_and_save(img0, img1, H, "./tmp/trt_warp_result_"+str(i)+".jpg")
        else:
            print("Failed to find H.")