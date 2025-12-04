cmake_minimum_required(VERSION 3.18)
project(RomaMatcher)

set(CMAKE_CXX_STANDARD 17)

# --- 配置路径 (请根据实际情况修改) ---
# LibTorch 路径
set(Torch_DIR "/home/percy/workspace/download/libtorch-2.9.1/libtorch/share/cmake/Torch") 
# TensorRT 安装路径
set(TENSORRT_DIR "/usr/local/TensorRT-10.8.0.43")
# OpenCV
find_package(OpenCV REQUIRED)
# CUDA
find_package(CUDAToolkit REQUIRED)

# --- 查找 LibTorch ---
find_package(Torch REQUIRED)

# --- 包含头文件 ---
include_directories(${TENSORRT_DIR}/include)
include_directories(src)

# --- 链接库 ---
# TensorRT 库
find_library(NVINFER_LIB nvinfer PATHS ${TENSORRT_DIR}/lib NO_DEFAULT_PATH)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS ${TENSORRT_DIR}/lib NO_DEFAULT_PATH) # 如果用到插件
find_library(NVONNXPARSER_LIB nvonnxparser PATHS ${TENSORRT_DIR}/lib NO_DEFAULT_PATH)

# --- 源文件 ---
file(GLOB SOURCES "src/*.cpp")

add_executable(roma_app ${SOURCES})

# --- 链接 ---
target_link_libraries(roma_app "${TORCH_LIBRARIES}")
target_link_libraries(roma_app ${OpenCV_LIBS})
target_link_libraries(roma_app ${NVINFER_LIB})
target_link_libraries(roma_app CUDA::cudart)

# 为了避免 CXX11 ABI 冲突，建议加上这个（取决于你下载的 LibTorch 版本）
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)