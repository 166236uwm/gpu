#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

__global__ void NaiveMM(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        C[row * N + col] = A[row * N + col] * B[row * N + col];
    }
}

void display_matrix(int n, float* A){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i * n + j];
        }
        std::cout<<std::endl;
    }
}

void zadanie1()
{
    std::cout << "Zadanie 1" << std::endl;
    const int N = 512;

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];
    float* h_C_gpu = new float[N * N];

    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = std::rand() % 10;
        h_B[i] = std::rand() % 10;
        h_C[i] = 0;
        h_C_gpu[i] = 0;
    }

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(block_size, block_size);

    dim3 numBlocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    NaiveMM << <numBlocks, threadsperblock >> > (d_A, d_B, d_C, N);

    cudaMemcpy(h_C_gpu, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << h_C << std::endl;
    std::cout << h_C_gpu << std::endl;

    // zwolnienie pamiêci CPU
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_gpu;

    // zwolnienie pamiêci GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main() {
    zadanie1();
    return 0;
}