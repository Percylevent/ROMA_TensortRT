import onnx
from onnxsim import simplify

# 加载刚才导出的模型
save_path ='./onnx/roma_core.onnx'
onnx_model = onnx.load(save_path)

# 执行简化
# dynamic_input_shape=True 对于动态分辨率模型很重要
model_simp, check = simplify(onnx_model) 

assert check, "Simplified ONNX model could not be validated"

# 覆盖保存
onnx.save(model_simp, save_path)
print(f"✅ Simplified model saved to {save_path}")