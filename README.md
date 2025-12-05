<p align="right">
  <a href="./README_zh.md"><strong>简体中文</strong></a>
</p>

# RoMa-TensorRT: Accelerate RoMa/GIM_ROMA by over 10x

[![GitHub Stars](https://img.shields.io/github/stars/Percylevent/ROMA_TensortRT.svg?style=social&label=Star)](https://github.com/Percylevent/ROMA_TensortRT) 

This project demonstrates the conversion of the [GIM_ROMA](https://github.com/xuelunshen/gim) and [ROMA](https://github.com/Parskatt/RoMa) image matching models from PyTorch to TensorRT for significant inference acceleration. Through optimization, we successfully reduced the inference time from **1.6 seconds** down to **0.1 seconds** (C++) on an NVIDIA RTX 4070 (Laptop).

This repository provides a complete workflow, including scripts for ONNX export, simplification, and example inference code in both Python and C++.

![Warp Result with TensorRT](./trt_warp_result.jpg "Warp Result with TensorRT")
*<p align="center">Warp result from Python TensorRT inference</p>*

![Warp Result with C++](./cpp_warp_result.jpg "Warp Result with C++")
*<p align="center">Warp result from C++ TensorRT inference</p>*

## 🌟 Project Highlights

- **Incredible Performance Boost**: Achieved over **16x** inference acceleration in C++.
- **End-to-End Workflow**: Provides a complete pipeline from PyTorch to ONNX and finally to a TensorRT engine.
- **Python & C++ Examples**: Includes inference code for both Python and C++, catering to different deployment needs.
- **Pre-converted Models**: Offers ready-to-use ONNX models to get you started quickly.

## 🛠️ Environment Setup

To ensure the code runs smoothly, please configure the following environment. Using a Conda virtual environment is highly recommended.

- Python 3.9+
- PyTorch
- ONNX
- ONNX-Simplifier
- ONNXRuntime-GPU
- OpenCV-Python

You can install the primary dependencies using pip:
```bash
pip install torch onnx onnx-simplifier onnxruntime-gpu opencv-python
```

**TensorRT Environment:**
- **Recommended**: TensorRT 10.8.0 (tested and stable).
- **Warning**: TensorRT 10.0.0 caused unexpected errors during our tests and is **not recommended**. Please download and install TensorRT from the official NVIDIA website and ensure its Python package is correctly installed.

## 🚀 Quick Start

### Step 1: Prepare the ONNX Model

You can either export the ONNX model yourself or download our pre-converted files.

#### Option A: Export the Model Yourself (Recommended)

1.  **Export the original ONNX model**:
    Ensure the model path in `extra.py` is correct, then run:
    ```bash
    python extra.py
    ```
    Upon success, you will find `roma_core.onnx` in the `onnx-ori` directory.

2.  **Simplify the ONNX model**:
    To achieve better performance, we simplify the model using `onnx-simplifier`.
    ```bash
    python onnxsimple.py
    ```
    The simplified model, `roma_core_sim.onnx`, will be saved in the `onnx-sim` directory.

#### Option B: Download Pre-converted Models

Alternatively, download the ONNX models from the links below and extract them to the project's root directory.

- [onnx-ori (Original Model)](https://drive.google.com/drive/folders/1--WvnclFGsjRBd_2ByscYWTKNusOJnrx?usp=sharing)
- [onnx-sim (Simplified Model)](https://drive.google.com/drive/folders/1DgLG-74HgarsDV1Inluuhodj6jF8hbdI?usp=sharing)

#### (Optional) Step 1.3: Test the ONNX Model
Before converting to TensorRT, you can verify the ONNX model's correctness.
```bash
python testonnx.py
```

### Step 2: Convert to a TensorRT Engine

The `.engine` file is hardware-specific, so you **must** generate it on your own machine.

Use the `trtexec` command-line tool to convert the ONNX model into a TensorRT engine. Remember to replace the paths with your own.

```bash
# Convert the onnx-ori model to an FP16 engine
trtexec --onnx=/path/to/your/roma_trt/onnx_ori/roma_core.onnx \
        --saveEngine=/path/to/your/roma_trt/roma_core_ori_fp16.engine \
        --minShapes=image_a:1x3x504x504,image_b:1x3x504x504 \
        --optShapes=image_a:1x3x504x504,image_b:1x3x504x504 \
        --maxShapes=image_a:1x3x504x504,image_b:1x3x504x504 \
        --fp16 --verbose
```
**Note**: Please delete the `.engine` files included in this repository before running, as they may not be compatible with your hardware.

### Step 3: Run TensorRT Inference

#### 3.1 Inference with Python
Once you have generated the `.engine` file, run the following command:
```bash
python testtrt.py
```

#### 3.2 Inference with C++
Our C++ example provides a reference for a production-like deployment.

1.  **Compile the code**:
    ```bash
    cd roma_cpp/build
    cmake ..
    make
    ```

2.  **Run the application**:
    Before running, ensure the `.engine` file is located at the correct path (the project root by default).
    ```bash
    ./roma_app
    ```

## 📊 Performance Comparison

The following results were benchmarked on an **NVIDIA RTX 4070 (Laptop)**. TensorRT provides a dramatic performance improvement.

| Platform          | Inference Time | Speedup |
|:------------------|:--------------:|:-------:|
| PyTorch           | ~1.6 s         | 1x      |
| TensorRT (Python) | ~0.16 s        | 10x     |
| TensorRT (C++)    | **~0.1 s**     | **16x** |

## 🤝 Contact the Author

Feel free to reach out if you have any questions or suggestions regarding this project.

- **Institution**: Shanghai Jiao Tong University
- **Email**: [zhangpengcheng@sjtu.edu.cn](mailto:zhangpengcheng@sjtu.edu.cn)
- **GitHub**: [https://github.com/Percylevent](https://github.com/Percylevent)

If you find this project helpful, please give it a ⭐️ Star!
