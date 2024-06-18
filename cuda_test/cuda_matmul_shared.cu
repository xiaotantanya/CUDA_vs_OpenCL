//计算矩阵乘法： C = A * B

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 512
#define K 512
#define N 512

void initial(float *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (float)(rand() % 10 + 1);
	}
}

//核函数（静态共享内存版）
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//分配共享内存
	__shared__ float sharedM[blockDim.y][blockDim.x];
	__shared__ float sharedN[blockDim.x][blockDim.y];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	float Csub = 0.0;
	
	//核心：下面将保存在全局内存中的矩阵M&N分块存放到共享内存中
	for (int i = 0; i < (int)(ceil((float)numAColumns / blockDim.x)); i++)//如上图，将一个红框矩形分成多个正方形
	{
		if (i*blockDim.x + tx < numAColumns && row < numARows)//分割M矩阵，边界确定方式结合上图蓝色正方形内数据的位置理解
			sharedM[ty][tx] = A[row*numAColumns + i * blockDim.x + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*blockDim.y + ty < numBRows && col < numBColumns)//分割N矩阵
			sharedN[ty][tx] = B[(i*blockDim.y + ty)*numBColumns + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();//同一线程块中所有线程必须到达运行 __synctrheads()之后才可以做其余操作
		//此操作可以防止当只有部分数据拷贝到共享内存后就提前进行下列计算。

		for (int j = 0; j < blockDim.x; j++)//分块后的矩阵相乘
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}

	if (row < numCRows && col < numCColumns)//将计算后的矩阵块放到结果矩阵C中
		C[row*numCColumns + col] = Csub;
}

//主函数（基本都是常规操作了，和普通版乘法差别不大）
int main(int argc, char **argv)
{
	int Axy = M * K;
	int Bxy = K * N;
	int Cxy = M * N;

	float *h_A, *h_B, *h_C, *deviceRef;
	h_A = (float*)malloc(Axy * sizeof(float));
	h_B = (float*)malloc(Bxy * sizeof(float));

	deviceRef = (float*)malloc(Cxy * sizeof(float));

	initial(h_A, Axy);
	initial(h_B, Bxy);
	
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, Axy * sizeof(float));
	cudaMalloc((void**)&d_B, Bxy * sizeof(float));
	cudaMalloc((void**)&d_C, Cxy * sizeof(float));

	cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);
	
    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	
	matrixMultiplyShared << < grid, block >> > (d_A, d_B, d_C, M, K, K, N, M, N);
	cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
	
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
}
