#include "RomaMatcher.h"

int main(int argc, char** argv) {
    // 路径配置 (可改为命令行参数)
    std::string engine_path = "../roma_core_ori_fp16.engine";
    std::string img0_path = "../1.jpg";
    std::string img1_path = "../2.jpg";

    if (argc >= 4) {
        engine_path = argv[1];
        img0_path = argv[2];
        img1_path = argv[3];
    }

    // 1. 初始化
    RomaMatcher matcher(engine_path, 504);

    // 2. 读取图片
    cv::Mat img0 = cv::imread(img0_path);
    cv::Mat img1 = cv::imread(img1_path);

    if (img0.empty() || img1.empty()) {
        std::cerr << "Error reading images." << std::endl;
        return -1;
    }

    // 3. 匹配计算 H
    for (int i = 0; i < 10; i++) {
        std::cout << "Calculating Homography..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat H = matcher.match(img0, img1);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Time taken: " << diff.count() << " s" << std::endl;

        std::cout << "Found H:\n" << H << std::endl;

        // 4. 可视化
        if (!H.empty()) {
            matcher.warp_and_save(img0, img1, H, "cpp_warp_result.jpg");
        }
        else {
            std::cout << "Failed to find valid Homography." << std::endl;
        }
    }
    return 0;
}