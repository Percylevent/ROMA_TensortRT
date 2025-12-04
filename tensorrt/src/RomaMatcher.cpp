#include "RomaMatcher.h"
#include <torch/nn/functional.h> // for pad, interpolate

using namespace torch::indexing;

RomaMatcher::RomaMatcher(const std::string& engine_path, int img_size)
    : m_img_size(img_size)
{
    // 初始化 CUDA Stream
    cudaStreamCreate(&m_stream);

    // 加载 Engine
    load_engine(engine_path);

    std::cout << "✅ TensorRT Engine Initialized (LibTorch Backend)!" << std::endl;
}

RomaMatcher::~RomaMatcher() {
    cudaStreamDestroy(m_stream);
    // TRT 指针由 shared_ptr 自动释放 (需自定义 deleter，这里简化处理，注意实际工程中最好用 unique_ptr + custom deleter)
    // 但由于 nvinfer1 的对象销毁顺序很重要，这里依赖 shared_ptr 的析构顺序
}

void RomaMatcher::load_engine(const std::string& engine_path) {
    m_logger = std::make_shared<TRTLogger>();
    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*m_logger));

    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) throw std::runtime_error("Error reading engine file");

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        m_runtime->deserializeCudaEngine(buffer.data(), size),
        [](nvinfer1::ICudaEngine* ptr) { if (ptr) delete ptr; } // TRT 8.x+ use delete, older use destroy()
    );

    if (!m_engine) throw std::runtime_error("Failed to deserialize engine");

    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(
        m_engine->createExecutionContext(),
        [](nvinfer1::IExecutionContext* ptr) { if (ptr) delete ptr; }
    );

    // 设置绑定
    int num_io = m_engine->getNbIOTensors();
    for (int i = 0; i < num_io; ++i) {
        const char* name = m_engine->getIOTensorName(i);
        nvinfer1::Dims dims = m_engine->getTensorShape(name);
        nvinfer1::DataType type = m_engine->getTensorDataType(name);
        nvinfer1::TensorIOMode mode = m_engine->getTensorIOMode(name);

        // 转换维度: 遇到 -1 处理为 1 (假设 Batch=1)
        std::vector<int64_t> torch_dims;
        for (int d = 0; d < dims.nbDims; ++d) {
            torch_dims.push_back(dims.d[d] == -1 ? 1 : dims.d[d]);
        }

        // 转换类型
        torch::ScalarType torch_type;
        if (type == nvinfer1::DataType::kFLOAT) torch_type = torch::kFloat32;
        else if (type == nvinfer1::DataType::kHALF) torch_type = torch::kHalf;
        else if (type == nvinfer1::DataType::kINT32) torch_type = torch::kInt32;
        else throw std::runtime_error("Unsupported dtype");

        // 分配显存
        torch::Tensor tensor = torch::empty(torch_dims, torch::TensorOptions().device(m_device).dtype(torch_type));

        Binding binding;
        binding.name = name;
        binding.index = i;
        binding.tensor = tensor;
        binding.is_input = (mode == nvinfer1::TensorIOMode::kINPUT);

        m_bindings.push_back(binding);
        m_tensor_name_to_ptr[name] = tensor.data_ptr();

        if (binding.is_input) {
            m_context->setInputShape(name, dims);
        }
    }
}

// 预处理
std::tuple<torch::Tensor, float, std::pair<int, int>> RomaMatcher::preprocess_official(const cv::Mat& img) {
    // BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    int h = rgb.rows;
    int w = rgb.cols;
    float scale = (float)m_img_size / std::max(h, w);

    int new_h = std::round(h * scale);
    int new_w = std::round(w * scale);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Normalize & HWC -> CHW
    torch::Tensor t = torch::from_blob(resized.data, { new_h, new_w, 3 }, torch::kByte);
    t = t.to(torch::kFloat32).div(255.0);
    t = t.permute({ 2, 0, 1 }).unsqueeze(0); // [1, 3, H, W]

    return { t, scale, {new_h, new_w} };
}

// 计算 Padding
std::tuple<int, int, int, int, int, int> RomaMatcher::get_padding_size(const torch::Tensor& image, int h, int w) {
    int orig_h = image.size(2);
    int orig_w = image.size(3);

    // 这里Python代码其实是用来计算 resize 后的尺寸的，但我们已经 resize 好了
    // 所以这里的逻辑简化为计算如何 pad 到目标 h, w
    // 注意：传入的 image 已经是 resize 后的

    int pad_height = h - orig_h;
    int pad_width = w - orig_w;

    int pad_top = pad_height / 2;
    int pad_bottom = pad_height - pad_top;
    int pad_left = pad_width / 2;
    int pad_right = pad_width - pad_left;

    return { orig_w, orig_h, pad_left, pad_right, pad_top, pad_bottom };
}

// KDE 实现
torch::Tensor RomaMatcher::kde(torch::Tensor x, float std) {
    x = x.to(torch::kHalf);
    // cdist: [N, 2] -> [N, N]
    auto scores = torch::exp(-torch::cdist(x, x).pow(2) / (2 * std * std));
    return scores.sum(-1); // [N]
}

// 后处理流程
std::pair<torch::Tensor, torch::Tensor> RomaMatcher::post_process_flow(
    torch::Tensor final_flow,
    torch::Tensor final_certainty,
    torch::Tensor low_res_certainty,
    torch::Tensor im_A,
    torch::Tensor im_B)
{
    int hs = m_img_size;
    int ws = m_img_size;

    torch::Tensor certainty = final_certainty;

    if (m_attenuate_cert) {
        auto low_res_interp = torch::nn::functional::interpolate(
            low_res_certainty,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{hs, ws})
            .align_corners(false)
            .mode(torch::kBilinear)
        );
        auto cert_clamp_mask = (low_res_interp < 0).to(torch::kFloat);
        auto attenuation = 0.5 * low_res_interp * cert_clamp_mask;
        certainty = certainty - attenuation;
    }

    certainty = torch::sigmoid(certainty);

    // 黑边抑制
    auto black_mask1 = (im_A < 0.03125).all(1, true); // [1, 1, H, W]
    auto black_mask2 = (im_B < 0.03125).all(1, true);

    auto chunks = certainty.chunk(2, 0);
    auto cert_a = torch::where(black_mask1, torch::tensor(0.0, m_device), chunks[0]);
    auto cert_b = torch::where(black_mask2, torch::tensor(0.0, m_device), chunks[1]);
    certainty = torch::cat({ cert_a, cert_b }, 0);

    // 网格生成
    auto grid_opts = torch::TensorOptions().device(m_device);
    auto grid_y_lin = torch::linspace(-1 + 1.0 / hs, 1 - 1.0 / hs, hs, grid_opts);
    auto grid_x_lin = torch::linspace(-1 + 1.0 / ws, 1 - 1.0 / ws, ws, grid_opts);
    auto grids = torch::meshgrid({ grid_y_lin, grid_x_lin }, "ij"); // y, x

    // stack: y, x -> x, y to match python stack(grid_x, grid_y)
    auto im_coords = torch::stack({ grids[1], grids[0] }, -1).unsqueeze(0).expand({ im_A.size(0), -1, -1, -1 });

    auto wrong_mask = (final_flow.abs() > 1).sum(1, true) > 0;
    certainty = torch::where(wrong_mask, torch::tensor(0.0, m_device), certainty);

    auto flow_clamped = torch::clamp(final_flow, -1, 1);
    auto flow_permuted = flow_clamped.permute({ 0, 2, 3, 1 }); // [2, H, W, 2]
    auto flow_chunks = flow_permuted.chunk(2, 0);

    auto warp_A = torch::cat({ im_coords, flow_chunks[0] }, -1);
    auto warp_B = torch::cat({ flow_chunks[1], im_coords }, -1);
    auto warp = torch::cat({ warp_A, warp_B }, 2); // [1, H, 2W, 2] ? No, python logic is cat dim=2, which is W dimension? 
    // Python: torch.cat((q_warp, s_warp), dim=2) where shape is [1, H, W, 4]
    // im_coords: [1, H, W, 2]
    // flow: [1, H, W, 2]
    // cat dim -1 -> [1, H, W, 4]
    // warp = cat dim 2?? Wait check python logic.
    // Python: warp = torch.cat((q_warp, s_warp), dim=2) -> This concatenates along Width.
    // But RoMa usually outputs dense matches for the whole image.
    // Let's stick to Python code:
    // q_warp = torch.cat((im_coords, A_to_B), dim=-1) -> [1, H, W, 4] (x1,y1, x2,y2)
    // s_warp = torch.cat((B_to_A, im_coords), dim=-1) -> [1, H, W, 4] (x2,y2, x1,y1) (inverse match)
    // warp = torch.cat((q_warp, s_warp), dim=2) -> [1, H, 2W, 4] ??? 
    // Usually RoMa returns [1, H, W, 4] for one direction.
    // Let's re-read Python: `warp = torch.cat((q_warp, s_warp), dim=2)`
    // dense_matches, dense_certainty returned.

    // Actually, dense_matches in python is `warp[0]`.
    // It seems it concatenates the forward matches and backward matches side-by-side?
    // Let's just follow the tensor ops exactly.
    warp = torch::cat({ warp_A, warp_B }, 2); // Along Width dimension

    auto cert_chunks = certainty.chunk(2, 0);
    auto certainty_final = torch::cat({ cert_chunks[0], cert_chunks[1] }, 3); // Along Width

    return { warp[0], certainty_final[0][0] };
}

// 采样
torch::Tensor RomaMatcher::sample(torch::Tensor dense_matches, torch::Tensor dense_certainty, int num, int factor) {
    // 固定随机种子
    //torch::manual_seed(0);
    //torch::cuda::manual_seed_all(0);

    auto certainty = dense_certainty.clone();
    certainty.index_put_({ certainty > m_sample_thresh }, 1.0);

    auto matches = dense_matches.reshape({ -1, 4 });
    certainty = certainty.reshape({ -1 });

    if (certainty.sum().item<float>() < 1e-6) {
        certainty += 1e-8;
    }

    int expansion_factor = factor;
    int num_samples = std::min((int)(expansion_factor * num), (int)certainty.size(0));

    auto good_samples_idx = torch::multinomial(certainty, num_samples, false);
    auto good_matches = matches.index({ good_samples_idx });

    auto density = kde(good_matches, 0.1f);
    auto p = 1.0 / (density + 1.0);
    p.index_put_({ density < 10 }, 1e-7);

    int final_num = std::min(num, (int)good_samples_idx.size(0));
    auto balanced_idx = torch::multinomial(p, final_num, false);

    return good_matches.index({ balanced_idx });
}

// 提取匹配点
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> RomaMatcher::extract_matches(const cv::Mat& img_A, const cv::Mat& img_B) {
    // 1. Preprocess
    auto res_A = preprocess_official(img_A);
    auto img0_t = std::get<0>(res_A);
    float scale0 = std::get<1>(res_A);
    auto size0 = std::get<2>(res_A); // h, w

    auto res_B = preprocess_official(img_B);
    auto img1_t = std::get<0>(res_B);
    float scale1 = std::get<1>(res_B);
    auto size1 = std::get<2>(res_B);

    // 2. Padding
    auto pad0 = get_padding_size(img0_t, m_img_size, m_img_size); // orig_w, orig_h, l, r, t, b
    auto pad1 = get_padding_size(img1_t, m_img_size, m_img_size);

    namespace F = torch::nn::functional;
    // Pad order in libtorch: {left, right, top, bottom}
    std::vector<int64_t> p0 = { std::get<2>(pad0), std::get<3>(pad0), std::get<4>(pad0), std::get<5>(pad0) };
    std::vector<int64_t> p1 = { std::get<2>(pad1), std::get<3>(pad1), std::get<4>(pad1), std::get<5>(pad1) };

    auto img0_pad = F::pad(img0_t.to(m_device), F::PadFuncOptions(p0)).contiguous();
    auto img1_pad = F::pad(img1_t.to(m_device), F::PadFuncOptions(p1)).contiguous();

    int h_pad = img0_pad.size(2);
    int w_pad = img0_pad.size(3);

    // 3. Inference
    // Find input bindings
    for (auto& b : m_bindings) {
        if (b.name.find("image_a") != std::string::npos || b.index == 0) { // Naive name check, better check index
            b.tensor.copy_(img0_pad);
        }
        else if (b.name.find("image_b") != std::string::npos || b.index == 1) {
            b.tensor.copy_(img1_pad);
        }
        m_context->setTensorAddress(b.name.c_str(), b.tensor.data_ptr());
    }

    // Execute Async V3
    m_context->enqueueV3(m_stream);
    cudaStreamSynchronize(m_stream);

    // 4. Retrieve outputs
    torch::Tensor final_flow, final_certainty, low_res_certainty;
    for (auto& b : m_bindings) {
        if (b.name == "final_flow") final_flow = b.tensor;
        else if (b.name == "final_certainty") final_certainty = b.tensor;
        else if (b.name == "low_res_certainty") low_res_certainty = b.tensor;
    }

    // 5. Post Process
    auto res = post_process_flow(final_flow, final_certainty, low_res_certainty, img0_pad, img1_pad);
    auto dense_matches = res.first;
    auto dense_certainty = res.second;

    // 6. Sample
    auto sparse_matches = sample(dense_matches, dense_certainty);

    // 7. Coordinate Restore
    auto kpts0_norm = sparse_matches.index({ Slice(), Slice(0, 2) });
    auto kpts1_norm = sparse_matches.index({ Slice(), Slice(2, 4) });

    // Map to padded size [0, 504]
    // (x+1)/2 * w
    auto kpts0 = torch::stack({
        w_pad * (kpts0_norm.index({Slice(), 0}) + 1) / 2,
        h_pad * (kpts0_norm.index({Slice(), 1}) + 1) / 2
        }, -1);

    auto kpts1 = torch::stack({
        w_pad * (kpts1_norm.index({Slice(), 0}) + 1) / 2,
        h_pad * (kpts1_norm.index({Slice(), 1}) + 1) / 2
        }, -1);

    // Subtract padding
    kpts0.index({ Slice(), 0 }) -= (float)std::get<2>(pad0); // left
    kpts0.index({ Slice(), 1 }) -= (float)std::get<4>(pad0); // top

    kpts1.index({ Slice(), 0 }) -= (float)std::get<2>(pad1); // left
    kpts1.index({ Slice(), 1 }) -= (float)std::get<4>(pad1); // top

    // Mask Filtering
    int vw0 = size0.second; int vh0 = size0.first;
    int vw1 = size1.second; int vh1 = size1.first;

    auto mask = (kpts0.index({ Slice(), 0 }) > 0) & (kpts0.index({ Slice(), 0 }) <= (vw0 - 1)) &
        (kpts0.index({ Slice(), 1 }) > 0) & (kpts0.index({ Slice(), 1 }) <= (vh0 - 1)) &
        (kpts1.index({ Slice(), 0 }) > 0) & (kpts1.index({ Slice(), 0 }) <= (vw1 - 1)) &
        (kpts1.index({ Slice(), 1 }) > 0) & (kpts1.index({ Slice(), 1 }) <= (vh1 - 1));

    kpts0 = kpts0.index({ mask });
    kpts1 = kpts1.index({ mask });

    // Un-scale
    kpts0 /= scale0;
    kpts1 /= scale1;

    // Convert to OpenCV Points
    std::vector<cv::Point2f> pts0, pts1;
    auto kpts0_cpu = kpts0.cpu();
    auto kpts1_cpu = kpts1.cpu();

    auto acc0 = kpts0_cpu.accessor<float, 2>();
    auto acc1 = kpts1_cpu.accessor<float, 2>();

    for (int i = 0; i < kpts0.size(0); ++i) {
        pts0.emplace_back(acc0[i][0], acc0[i][1]);
        pts1.emplace_back(acc1[i][0], acc1[i][1]);
    }

    return { pts0, pts1 };
}

cv::Mat RomaMatcher::match(const cv::Mat& img_A, const cv::Mat& img_B) {
    auto pts = extract_matches(img_A, img_B);
    auto& ptsA = pts.first;
    auto& ptsB = pts.second;

    if (ptsA.size() < 8) {
        std::cerr << "Not enough matches found." << std::endl;
        return cv::Mat();
    }

    cv::Mat H = cv::findHomography(ptsA, ptsB, cv::USAC_MAGSAC, 4.0, cv::noArray(), 10000, 0.99999);
    return H;
}

void RomaMatcher::warp_and_save(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& H, const std::string& save_path) {
    if (H.empty()) return;

    cv::Mat warped;
    cv::warpPerspective(img0, warped, H, img1.size());

    cv::Mat blend;
    cv::addWeighted(img1, 0.5, warped, 0.5, 0, blend);

    cv::imwrite(save_path, blend);
    std::cout << "✅ Warp visualization saved to " << save_path << std::endl;
}