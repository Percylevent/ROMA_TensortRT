# export_roma_core.py

import torch
import torch.nn as nn
import os
import numpy as np
from roma import RoMa 
import onnx
import onnxruntime as ort
from onnxsim import simplify


def fuse_roma_model(model):
    print("⚡ Fusing BatchNormalization layers for TensorRT optimization...")
    
    # 遍历 model.decoder.conv_refiner 中的所有 scale
    # conv_refiner 是一个 ModuleDict
    for scale, refiner in model.decoder.conv_refiner.items():
        if hasattr(refiner, 'fuse'):
            print(f"   - Fusing ConvRefiner scale {scale}")
            refiner.fuse()
            
class RomaCoreOnnxWrapper(nn.Module):
    """
    这个包装器只封装 RoMa 模型的核心神经网络计算。
    """
    def __init__(self, roma_model: nn.Module):
        super().__init__()
        self.model = roma_model

    def forward(self, image_a, image_b):
        # 直接调用我们为 ONNX 导出的核心方法
        return self.model.forward_core_for_onnx(image_a, image_b)

def export_core_roma_to_onnx(ckpt_path, save_path, processing_size=504):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading full RoMa model...")
    # 确保 RoMa 初始化时，所有子模块都已是静态版本
    full_model = RoMa(img_size=[processing_size])
    
    try:
        loaded_data = torch.load(ckpt_path, map_location='cpu')
        state_dict = loaded_data.get('state_dict', loaded_data)
        weights = {k.replace("model.", ""): v for k, v in state_dict.items()}
        # 注意：这里需要加载到 RegressionMatcher 实例中
        full_model.load_state_dict(weights, strict=False)
        full_model.eval()
        fuse_roma_model(full_model)
        full_model.to(device)
        print("Full RoMa model loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print("Instantiating ONNX wrapper for core computation...")
    onnx_wrapper = RomaCoreOnnxWrapper(full_model).to(device).eval()

    # --- 准备导出 ---
    batch_size = 1
    dummy_img_a = torch.randn(batch_size, 3, processing_size, processing_size, device=device)
    dummy_img_b = torch.randn(batch_size, 3, processing_size, processing_size, device=device)
    
    input_names = ['image_a', 'image_b']
    output_names = ['final_flow', 'final_certainty', 'low_res_certainty']
    dynamic_axes = {
        'image_a': {0: 'batch_size'}, 'image_b': {0: 'batch_size'},
        'final_flow': {0: 'batch_size'}, 'final_certainty': {0: 'batch_size'},
        'low_res_certainty': {0: 'batch_size'},
    }
    
    # --- 导出到 ONNX ---
    print(f"Exporting core RoMa model to ONNX at {save_path}...")
    try:
        torch.onnx.export(
            onnx_wrapper, (dummy_img_a, dummy_img_b), save_path,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=17,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True, verbose=False
        )
        print("✅ ONNX export successful!")
    except Exception as e:
        print(f"❌ An error occurred during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 验证 ---
    print("\n--- Verifying the exported ONNX model ---")
    try:
        ort_session = ort.InferenceSession(save_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        with torch.no_grad():
            torch_outputs = onnx_wrapper(dummy_img_a, dummy_img_b)
            
        ort_inputs = {'image_a': dummy_img_a.cpu().numpy(), 'image_b': dummy_img_b.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        for i, name in enumerate(output_names):
            np.testing.assert_allclose(torch_outputs[i].cpu().numpy(), ort_outputs[i], rtol=1e-3, atol=1e-4)
            print(f"✅ Output '{name}' matches.")
        
        print("✅ ONNX model verification successful!")
    except Exception as e:
        print(f"❌ An error occurred during ONNX verification: {e}")

if __name__ == '__main__':
    CHECKPOINT_PATH = "./gim_roma_100h.ckpt" 
    ONNX_SAVE_DIR = "./onnx/"
    os.makedirs(ONNX_SAVE_DIR, exist_ok=True)
    onnx_save_path = os.path.join(ONNX_SAVE_DIR, "roma_core.onnx")
    export_core_roma_to_onnx(CHECKPOINT_PATH, onnx_save_path)