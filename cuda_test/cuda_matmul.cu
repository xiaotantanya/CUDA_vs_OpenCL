#include <iostream>


// # define M 5120
// # define K 5120
// # define N 5120

void initial(float *array, int size){
    for (int i = 0; i < size; i++){
        array[i] = (float)(rand() % 10 + 1);
    }
}

__global__ void multiplicateMatrix(float *array_A, float *array_B, float *array_C, int n){
    int ix = threadIdx.x + blockDim.x*blockIdx.x;
    int iy = threadIdx.y + blockDim.y*blockIdx.y;

    if (ix < n && iy < n) {
        float sum = 0;
        for(int k = 0; k < n; k++){
            sum += array_A[iy*n + k] * array_B[k*n + ix];
        }
        array_C[iy*n + ix] = sum;
    }
}

int main(int argc, char **argv){
    // 检查传递的参数数量是否正确
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <integer> " << std::endl;
        return 1;
    }
    int n = std::atoi(argv[1]);
    int Axy = n * n;
    int Bxy = n * n;
    int Cxy = n * n;

    float *h_A, *h_B, *h_C, *hostRef, *deviceRef;

    h_A = (float*)malloc(Axy * sizeof(float));
    h_B = (float*)malloc(Bxy * sizeof(float));
    h_C = (float*)malloc(Cxy * sizeof(float));

    initial(h_A, Axy);
    initial(h_B, Bxy);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, Axy*sizeof(float));
    cudaMalloc((void**)&d_B, Bxy*sizeof(float));
    cudaMalloc((void**)&d_C, Cxy*sizeof(float));

    cudaMemcpy(d_A, h_A, Axy*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bxy*sizeof(float), cudaMemcpyHostToDevice);

    int dimx = 2;
    int dimy = 2;
    dim3 block(dimx, dimy);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    // 记录 CUDA 内核的开始和结束时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    multiplicateMatrix<<<grid,block>>> (d_A, d_B, d_C, n);
    
    // 记录 CUDA 内核的结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出执行时间
    std::cout << "Matrix multiplication completed in " << milliseconds << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}