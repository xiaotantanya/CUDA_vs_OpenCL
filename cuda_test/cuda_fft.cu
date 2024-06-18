#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <chrono>

#define TRANSFORM_SIZE 2048 // 单个傅里叶变换的大小

// 错误检查宏
#define CHECK_CUDA_ERROR(call) {                                 \
    cudaError_t err = call;                                      \
    if (err != cudaSuccess) {                                    \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
}

#define CHECK_CUFFT_ERROR(call) {                                \
    cufftResult err = call;                                      \
    if (err != CUFFT_SUCCESS) {                                  \
        std::cerr << "cuFFT Error: " << err << std::endl;        \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
}

int main(int argc, char** argv) {
    // 检查传递的参数数量是否正确
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <integer> " << std::endl;
        return 1;
    }
    int NUM_TRANSFORMS = std::atoi(argv[1]);
    
    // 分配主机内存
    size_t total_size = NUM_TRANSFORMS * TRANSFORM_SIZE * sizeof(cufftComplex);
    cufftComplex* h_data = (cufftComplex*)malloc(total_size);

    // 初始化输入数据
    for (int i = 0; i < NUM_TRANSFORMS * TRANSFORM_SIZE; ++i) {
        h_data[i].x = 1.0f; // 实部
        h_data[i].y = 0.0f; // 虚部
    }

    // 分配设备内存
    cufftComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, total_size));

    // 将输入数据从主机内存复制到设备内存
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice));

    // 创建 cuFFT 计划
    cufftHandle plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, TRANSFORM_SIZE, CUFFT_C2C, NUM_TRANSFORMS));

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 执行傅里叶变换
    CHECK_CUFFT_ERROR(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    // 记录结束时间
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;

    // 将结果从设备内存复制到主机内存
    CHECK_CUDA_ERROR(cudaMemcpy(h_data, d_data, total_size, cudaMemcpyDeviceToHost));

    // 输出执行时间
    std::cout << "Time to perform " << NUM_TRANSFORMS << " FFTs: " << duration.count() << " ms" << std::endl;

    // 释放资源
    CHECK_CUFFT_ERROR(cufftDestroy(plan));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    free(h_data);

    return 0;
}