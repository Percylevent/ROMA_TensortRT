# RoMa-TensorRT: 将 RoMa/GIM_ROMA 模型加速10倍以上

[![GitHub Stars](https://img.shields.io/github/stars/Percylevent/ROMA_TensortRT.svg?style=social&label=Star)](https://github.com/Percylevent/ROMA_TensortRT)

这是一个将 PyTorch 实现的 [GIM_ROMA](https://github.com/xuelunshen/gim) 和 [ROMA](https://github.com/Parskatt/RoMa) 图像匹配模型转换为 TensorRT 的项目，旨在实现显著的推理加速。通过优化，我们成功在 NVIDIA RTX 4070 (Laptop) 平台上将推理时间从 **1.6秒** 缩短至 **0.1秒** (C++)。

这个仓库提供了完整的 ONNX 导出、优化以及 TensorRT 推理的 Python 和 C++ 示例代码。

![Warp Result with TensorRT](./trt_warp_result.jpg "Warp Result with TensorRT")
*<p align="center">Python TensorRT 推理效果图</p>*

![Warp Result with C++](./cpp_warp_result.jpg "Warp Result with C++")
*<p align="center">C++ TensorRT 推理效果图</p>*

## 🌟 项目亮点

- **惊人的性能提升**: 在 C++ 中实现了超过 **16倍** 的推理加速。
- **完整的实现流程**: 提供了从 PyTorch 到 ONNX，再到 TensorRT engine 的完整转换脚本。
- **Python & C++ 双语示例**: 同时提供了 Python 和 C++ 的 TensorRT 推理代码，满足不同部署需求。
- **预训练模型**: 直接提供转换好的 ONNX 模型，方便快速上手。

## 🛠️ 环境配置

为了保证代码的顺利运行，请配置以下环境。建议使用 Conda 创建虚拟环境。

- Python 3.9+
- PyTorch
- ONNX
- ONNX-Simplifier
- ONNXRuntime-GPU
- OpenCV-Python

你可以通过以下命令安装主要依赖：

```bash
pip install torch onnx onnx-simplifier onnxruntime-gpu opencv-python
```

**TensorRT 环境:**

- **强烈推荐**: TensorRT 10.8.0 (经测试稳定)。
- **注意**: TensorRT 10.10.0 在测试中遇到未知错误，不推荐使用。请从 NVIDIA 官网下载并安装 TensorRT，并确保其 Python 包已正确安装。

## 🚀 快速开始

### 第1步: 准备 ONNX 模型

你可以选择自己导出 ONNX 模型，或者直接下载我们已经转换好的模型。

#### 选项 A: 自己导出模型 (推荐)

1.  **导出原始 ONNX 模型**:
    首先下载[gim_roma.ckpt](https://drive.google.com/file/d/1j9aCfzhMlgLuoSNYaDXmHbVoJTIkK6xh/view?usp=sharing)权重文件，并放到/pytorch路径下。  
    请确保 `extra.py` 文件中的模型路径正确，然后运行：
    ```bash
    python extra.py
    ```
    运行成功后，你将在 `onnx-ori` 目录下得到 `roma_core.onnx` 文件。

3.  **简化 ONNX 模型**:
    为了获得更好的性能，我们使用 `onnx-simplifier` 对模型进行简化。
    ```bash
    python onnxsimple.py
    ```
    简化后的模型 `roma_core_sim.onnx` 将保存在 `onnx-sim` 目录下。

#### 选项 B: 直接下载

你也可以从下面的链接直接下载我们准备好的 ONNX 模型，并解压到项目根目录。
- [onnx_model](https://drive.google.com/drive/folders/1ehF6EUMwra4uHAFHjw6YR1VGwEqi4ZJm?usp=sharing)  
- [onnx-ori (原始模型)]
- [onnx-sim (简化模型)]

#### 步骤 1.3 (可选): 测试 ONNX 模型

在进行 TensorRT 转换之前，你可以先测试 ONNX 模型的正确性。

```bash
python testonnx.py```

### 第2步: 转换为 TensorRT Engine

`.engine` 文件是与硬件相关的，你 **必须** 在你自己的机器上生成它。

使用 `trtexec` 工具将 ONNX 模型转换为 TensorRT engine。请根据你的实际路径修改以下命令。

```bash
# 将 onnx-ori 模型转换为 FP16 engine
trtexec --onnx=/path/to/your/roma_trt/onnx_ori/roma_core.onnx \
        --saveEngine=/path/to/your/roma_trt/roma_core_ori_fp16.engine\
        --minShapes=image_a:1x3x504x504,image_b:1x3x504x504 \
        --optShapes=image_a:1x3x504x504,image_b:1x3x504x504 \
        --maxShapes=image_a:1x3x504x504,image_b:1x3x504x504 \
        --fp16 --verbose
```

**注意**: 在运行前，请删除本仓库中自带的 `.engine` 文件，因为它们在你的机器上可能无法工作。

### 第3步: 运行 TensorRT 推理

#### 3.1 使用 Python 进行推理

确保你已经生成了 `.engine` 文件，然后运行：

```bash
python testtrt.py
```

#### 3.2 使用 C++ 进行推理

我们的 C++ 示例提供了一个更接近生产环境的部署参考。

1.  **编译**:
    ```bash
    cd roma_cpp/build
    cmake ..
    make
    ```

2.  **运行**:
    在运行前，请确保 `.engine` 文件位于正确的路径（默认为项目根目录）。
    ```bash
    ./roma_app
    ```

## 📊 性能对比

在 **NVIDIA RTX 4070 (Laptop)** 上的测试结果如下。TensorRT 带来了显著的性能提升。

| 平台 | 推理时间 | 加速比 |
|:---:|:---:|:---:|
| PyTorch | ~1.6 s | 1x |
| TensorRT (Python) | ~0.16 s | 10x |
| TensorRT (C++) | **~0.1 s** | **16x** |

## 🤝 联系作者

如果你对这个项目有任何疑问或建议，欢迎随时联系我。

- **Institution**: Shanghai Jiao Tong University
- **Email**: [zhangpengcheng@sjtu.edu.cn](mailto:zhangpengcheng@sjtu.edu.cn)
- **GitHub**: [https://github.com/Percylevent](https://github.com/Percylevent)

如果你觉得这个项目对你有帮助，请给一个 ⭐️ Star！
