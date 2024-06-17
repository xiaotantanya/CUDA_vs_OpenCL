#include <iostream>
#include <cufft.h>
#include <chrono>

using namespace std;

void printComplexArray(cufftComplex *data, int n) {
  for (int i = 0; i < n; i++) {
    printf("%f + %fi\n", data[i].x, data[i].y);
  }
}

int main() {
  // 设置数据大小
  const int n = 5000;

  // 分配内存
  cufftComplex *data_in, *data_out;
  cufftPlan plan;

  cudaMalloc(&data_in, n * sizeof(cufftComplex));
  cudaMalloc(&data_out, n * sizeof(cufftComplex));

  // 初始化数据
  for (int i = 0; i < n; i++) {
    data_in[i].x = rand() / (float)RAND_MAX;
    data_in[i].y = rand() / (float)RAND_MAX;
  }

  // 创建傅里叶变换计划
  cufftPlanCreate(&plan, n, CUFFT_TYPE_COMPLEX, CUFFT_FORWARD);

  // 记录开始时间
  auto start = chrono::high_resolution_clock::now();

  // 执行傅里叶变换
  cufftExec(plan, data_in, data_out);

  // 记录结束时间
  auto end = chrono::high_resolution_clock::now();

  // 计算运行时间
  chrono::duration<double> elapsed = end - start;
  double run_time = elapsed.count();

  // 打印运行时间
  printf("运行时间: %f秒\n", run_time);

  // 释放内存
  cudaFree(data_in);
  cudaFree(data_out);
  cufftPlanDestroy(plan);

  return 0;
}
