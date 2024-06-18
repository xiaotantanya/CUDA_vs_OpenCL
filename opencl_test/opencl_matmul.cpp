#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>


void initial(float *array, int size){
    for (int i = 0; i < size; i++){
        array[i] = (float)(rand() % 10 + 1);
    }
}

const char* kernelSource = R"(
__kernel void matMul(__global float* A, __global float* B, __global float* C, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
)";

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << err << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    // 检查传递的参数数量是否正确
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <integer> " << std::endl;
        return 1;
    }
    // Matrix dimensions
    const int N = std::atoi(argv[1]);
    size_t grid_N = static_cast<size_t>(N);

    int Axy = N * N;
    int Bxy = N * N;
    int Cxy = N * N;

    float *A, *B, *C;
    A = (float*)malloc(Axy * sizeof(float));
    B = (float*)malloc(Bxy * sizeof(float));
    C = (float*)malloc(Cxy * sizeof(float));

    initial(A, Axy);
    initial(B, Bxy);

    // Get platforms and devices
    cl_platform_id platform;
    cl_device_id device;
    cl_uint numPlatforms, numDevices;
    checkError(clGetPlatformIDs(1, &platform, &numPlatforms), "clGetPlatformIDs");
    checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices), "clGetDeviceIDs");

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // Create buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, A, nullptr);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, B, nullptr);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N, nullptr, nullptr);

    // Create and build program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    checkError(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram");

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "matMul", nullptr);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // Define global and local work sizes
    size_t globalWorkSize[2] = {grid_N, grid_N};
    
    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Enqueue kernel execution
    checkError(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");

    // Read results from the device
    checkError(clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * N * N, C, 0, nullptr, nullptr), "clEnqueueReadBuffer");

    // End measuring time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix multiplication completed in " << 1000 * elapsed.count() << " ms." << std::endl;

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
