#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <chrono>

// OpenCV
#include <opencv2/opencv.hpp>

// LibTorch
#include <torch/torch.h>
#include <torch/script.h>

// TensorRT
#include <NvInfer.h>

// 简单的 TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

class RomaMatcher {
public:
    RomaMatcher(const std::string& engine_path, int img_size = 504);
    ~RomaMatcher();

    // 核心接口
    cv::Mat match(const cv::Mat& img_A, const cv::Mat& img_B);

    // 可视化辅助
    static void warp_and_save(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& H, const std::string& save_path);

private:
    // TensorRT 相关
    std::shared_ptr<TRTLogger> m_logger;
    std::shared_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;

    // 资源管理
    cudaStream_t m_stream;

    // 输入输出绑定信息
    struct Binding {
        std::string name;
        int index;
        torch::Tensor tensor; // 使用 Torch Tensor 管理 GPU 内存
        bool is_input;
    };
    std::vector<Binding> m_bindings;
    std::map<std::string, void*> m_tensor_name_to_ptr;

    int m_img_size;
    float m_sample_thresh = 0.05f;
    bool m_attenuate_cert = true;
    torch::Device m_device = torch::kCUDA;

private:
    // 内部辅助函数
    void load_engine(const std::string& engine_path);

    std::tuple<torch::Tensor, float, std::pair<int, int>> preprocess_official(const cv::Mat& img);

    std::tuple<int, int, int, int, int, int> get_padding_size(const torch::Tensor& image, int h, int w);

    torch::Tensor kde(torch::Tensor x, float std = 0.1f);

    std::pair<torch::Tensor, torch::Tensor> post_process_flow(
        torch::Tensor final_flow,
        torch::Tensor final_certainty,
        torch::Tensor low_res_certainty,
        torch::Tensor im_A,
        torch::Tensor im_B
    );

    torch::Tensor sample(torch::Tensor dense_matches, torch::Tensor dense_certainty, int num = 2000, int factor = 4);

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> extract_matches(const cv::Mat& img_A, const cv::Mat& img_B);
};