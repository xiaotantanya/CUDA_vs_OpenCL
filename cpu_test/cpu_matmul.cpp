#include <iostream>
#include <chrono>

// # define M 5120
// # define K 5120
// # define N 5120

void initial(float *array, int size){
    for (int i = 0; i < size; i++){
        array[i] = (float)(rand() % 10 + 1);
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

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0;i<n;i++){
        float sum = 0;
        for(int j=0;j<n;j++){
            for(int k=0;k<n;k++){
                sum += h_A[i*n+k] * h_B[k*n+j];
            }
            h_C[n*i+j] = sum;
        }
    }

    // End measuring time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Matrix multiplication completed in " << 1000 * elapsed.count() << " ms." << std::endl;

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}