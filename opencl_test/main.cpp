#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>

const char* multiplySource = R"(
__kernel void vec_add(__global const float *a,
                      __global const float *b,
                      __global float *c,
                      const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] * b[gid];
    }
}
)";

const char* addSource = R"(
__kernel void vec_add(__global const float *a,
                      __global const float *b,
                      __global float *c,
                      const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
)";

const char* subSource = R"(
__kernel void vec_add(__global const float *a,
                      __global const float *b,
                      __global float *c,
                      const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] - b[gid];
    }
}
)";

const char* devideSource = R"(
__kernel void vec_add(__global const float *a,
                      __global const float *b,
                      __global float *c,
                      const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] / b[gid];
    }
}
)";

int main(int argc, char **argv) {
    // 检查传递的参数数量是否正确
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <integer> <character>" << std::endl;
        return 1;
    }
    // 定义数组大小
    const unsigned int n = std::atoi(argv[1]);
    char* operation = argv[2];

    // 创建输入数据
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n);

    // 获取所有可用平台
    cl_uint numPlatforms;
    cl_platform_id platform = nullptr;
    cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Getting Platforms. (clGetPlatformsIDs)\n";
        return 1;
    }

    if (numPlatforms > 0) {
        std::vector<cl_platform_id> platforms(numPlatforms);
        status = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        platform = platforms[0]; // 选择第一个平台
    }

    // 获取设备
    cl_uint numDevices = 0;
    cl_device_id device = nullptr;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (numDevices > 0) {
        std::vector<cl_device_id> devices(numDevices);
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        device = devices[0]; // 选择第一个设备
    }

    // 创建上下文
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);

    // 创建命令队列
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);

    // 创建缓冲区
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data(), &status);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data(), &status);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &status);

    cl_program program;
    // 创建并编译程序
    if(strcmp(operation, "multiply") == 0){
        program = clCreateProgramWithSource(context, 1, &multiplySource, nullptr, &status);
    }else if(strcmp(operation, "devide") == 0){
        program = clCreateProgramWithSource(context, 1, &devideSource, nullptr, &status);
    }else if(strcmp(operation, "add") == 0){
        program = clCreateProgramWithSource(context, 1, &addSource, nullptr, &status);
    }else if(strcmp(operation, "sub") == 0){
        program = clCreateProgramWithSource(context, 1, &subSource, nullptr, &status);
    }else{
        throw std::invalid_argument("Not implemented !");
    }

    status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        // 打印编译错误日志
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Error in kernel: " << log.data() << "\n";
        return 1;
    }

    // 创建内核
    cl_kernel kernel = clCreateKernel(program, "vec_add", &status);

    // 设置内核参数
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    // 定义全局工作项大小
    size_t globalSize = n;

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 执行内核
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // 读取结果
    clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, sizeof(float) * n, c.data(), 0, nullptr, nullptr);

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 打印运行时间
    std::cout << "Time to perform 100,000,000 multiplications: " << 1000*elapsed.count() << " ms\n";

    // 验证结果
    bool correct = true;
    if(strcmp(operation, "multiply") == 0){
        for (size_t i = 0; i < n; ++i) {
            if (c[i] != a[i] * b[i]) {
                correct = false;
                break;
            }
        }
    }else if(strcmp(operation, "devide") == 0){
        for (size_t i = 0; i < n; ++i) {
            if (c[i] != a[i] / b[i]) {
                correct = false;
                break;
            }
        }
    }else if(strcmp(operation, "add") == 0){
        for (size_t i = 0; i < n; ++i) {
            if (c[i] != a[i] + b[i]) {
                correct = false;
                break;
            }
        }
    }else if(strcmp(operation, "sub") == 0){
        for (size_t i = 0; i < n; ++i) {
            if (c[i] != a[i] - b[i]) {
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
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
