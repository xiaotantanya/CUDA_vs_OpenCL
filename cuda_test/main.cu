#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

#define NUM_ELEMENTS 100000000

int main() {
    // 分配主机内存
    cufftComplex* h_data = (cufftComplex*)malloc(sizeof(cufftComplex) * NUM_ELEMENTS);

    // 分配设备内存
    cufftComplex* d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * NUM_ELEMENTS);

    // 创建 cuFFT 计划
    cufftHandle plan;
    cufftPlan1d(&plan, NUM_ELEMENTS, CUFFT_C2C, 1);

    // 将输入数据从主机内存复制到设备内存
    cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * NUM_ELEMENTS, cudaMemcpyHostToDevice);

    // 执行傅里叶变换
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // 将结果从设备内存复制到主机内存
    cudaMemcpy(h_data, d_data, sizeof(cufftComplex) * NUM_ELEMENTS, cudaMemcpyDeviceToHost);

    // 释放资源
    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
