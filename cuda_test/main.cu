#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

// CUDA 内核函数：执行乘法运算
__global__ void multiplyKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// CUDA 内核函数：执行加法运算
__global__ void addKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA 内核函数：执行减法运算
__global__ void subKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

// CUDA 内核函数：执行除法运算
__global__ void devideKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

int main(int argc, char** argv) {
    // 检查传递的参数数量是否正确
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <integer> <character>" << std::endl;
        return 1;
    }
    // 设置问题的操作和规模
    const int numElements = std::atoi(argv[1]);
    char* operation = argv[2];

    const size_t size = numElements * sizeof(float);

    // 分配主机内存
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    // 初始化输入数据
    for (int i = 0; i < numElements; ++i) {
        h_a[i] = 1.0f;  // 或者其他值
        h_b[i] = 2.0f;  // 或者其他值
    }

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 将输入数据从主机内存复制到设备内存
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 配置线程块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // 记录 CUDA 内核的开始和结束时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 执行 CUDA 内核
    if(strcmp(operation, "multiply") == 0){
        multiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
    }else if(strcmp(operation, "devide") == 0){
        devideKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
    }else if(strcmp(operation, "add") == 0){
        addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
    }else if(strcmp(operation, "sub") == 0){
        subKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
    }else{
        // std::cout << operation << std::endl;
        throw std::invalid_argument("Not implemented !");
    }

    // 记录 CUDA 内核的结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 将结果从设备内存复制到主机内存
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 输出执行时间
    std::cout << "Time to perform 100,000,000 multiplications: " << milliseconds << " ms" << std::endl;

    bool correct = true;
    if(strcmp(operation, "multiply") == 0){
        for (size_t i = 0; i < numElements; ++i) {
            if (h_c[i] != h_a[i] * h_b[i]) {
                correct = false;
                break;
            }
        }
    }else if(strcmp(operation, "devide") == 0){
        for (size_t i = 0; i < numElements; ++i) {
            if (h_c[i] != h_a[i] / h_b[i]) {
                correct = false;
                break;
            }
        }
    }else if(strcmp(operation, "add") == 0){
        for (size_t i = 0; i < numElements; ++i) {
            if (h_c[i] != h_a[i] + h_b[i]) {
                correct = false;
                break;
            }
        }
    }else if(strcmp(operation, "sub") == 0){
        for (size_t i = 0; i < numElements; ++i) {
            if (h_c[i] != h_a[i] - h_b[i]) {
                correct = false;
                break;
            }
        }
    }else{
        // std::cout << operation << std::endl;
        throw std::invalid_argument("Not implemented !");
    }
    
    if (correct) {
        std::cout << "Result right! \n";
    } else {
        std::cout << "Result false! \n";
    }
    // 释放资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
