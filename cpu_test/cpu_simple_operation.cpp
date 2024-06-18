#include <iostream>
#include <cstring>
#include <chrono>

void add(float* a, float* b, float* c, int num){
    for(int i=0;i<num;i++){
        c[i] = a[i] + b[i];
    }
}

void sub(float* a, float* b, float* c, int num){
    for(int i=0;i<num;i++){
        c[i] = a[i] - b[i];
    }
}

void multiply(float* a, float* b, float* c, int num){
    for(int i=0;i<num;i++){
        c[i] = a[i] * b[i];
    }
}

void devide(float* a, float* b, float* c, int num){
    for(int i=0;i<num;i++){
        c[i] = a[i] / b[i];
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

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    // // 记录 CUDA 内核的开始和结束时间
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);

    // 执行 CUDA 内核
    if(strcmp(operation, "multiply") == 0){
        multiply(h_a, h_b, h_c, numElements);
    }else if(strcmp(operation, "devide") == 0){
        devide(h_a, h_b, h_c, numElements);
    }else if(strcmp(operation, "add") == 0){
        add(h_a, h_b, h_c, numElements);
    }else if(strcmp(operation, "sub") == 0){
        sub(h_a, h_b, h_c, numElements);
    }else{
        // std::cout << operation << std::endl;
        throw std::invalid_argument("Not implemented !");
    }

    // // 记录 CUDA 内核的结束时间
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // // 计算执行时间
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);


    // 输出执行时间
    std::cout << "Time to perform " << numElements <<" "<< operation << ": " << 1000*elapsed.count() << " ms" << std::endl;

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
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
