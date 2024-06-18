#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// FFT kernel (simplified, non-optimized version)
const char* kernelSource = R"(
__kernel void fft(__global float2* data, int N) {
    int gid = get_global_id(0);
    int num_transforms = get_global_size(0) / N;

    for (int t = 0; t < num_transforms; t++) {
        int start_idx = t * N;
        for (int s = 1; s < N; s <<= 1) {
            float theta = -M_PI / s;
            float2 W = (float2)(cos(theta), sin(theta));

            for (int k = 0; k < s; ++k) {
                float2 twiddle = (float2)(1.0, 0.0);
                for (int j = k; j < N; j += 2 * s) {
                    int idx = start_idx + j;
                    int idx_s = idx + s;

                    float2 t = data[idx_s] * twiddle;
                    data[idx_s] = data[idx] - t;
                    data[idx] = data[idx] + t;

                    twiddle = (float2)(twiddle.x * W.x - twiddle.y * W.y,
                                       twiddle.x * W.y + twiddle.y * W.x);
                }
            }
        }
    }
}
)";

int main() {
    const int NUM_TRANSFORMS = 1024;
    const int TRANSFORM_SIZE = 204800;

    size_t total_size = NUM_TRANSFORMS * TRANSFORM_SIZE;

    std::vector<cl_float2> h_data(total_size);
    for (int i = 0; i < total_size; ++i) {
        h_data[i] = {1.0f, 0.0f};
    }

    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context({device});
    cl::CommandQueue queue(context, device);

    cl::Buffer d_data(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * total_size);

    queue.enqueueWriteBuffer(d_data, CL_TRUE, 0, sizeof(cl_float2) * total_size, h_data.data());

    cl::Program program(context, kernelSource);
    program.build({device});

    cl::Kernel kernel(program, "fft");
    kernel.setArg(0, d_data);
    kernel.setArg(1, TRANSFORM_SIZE);

    cl::NDRange global(total_size);
    cl::NDRange local(TRANSFORM_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    queue.finish();
    
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;

    queue.enqueueReadBuffer(d_data, CL_TRUE, 0, sizeof(cl_float2) * total_size, h_data.data());

    std::cout << "Time to perform " << NUM_TRANSFORMS << " FFTs: " << duration.count() << " ms" << std::endl;

    return 0;
}
