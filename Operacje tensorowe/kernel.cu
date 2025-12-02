#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#define TILE_SIZE 16
__global__ void NaiveMM(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int idx = row * N + col;
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void MM(const float* A, const float* B, float* C, int N)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;

    sA[threadIdx.y][threadIdx.x] = A[row * N + col];
    sB[threadIdx.y][threadIdx.x] = B[row * N + col];

    __syncthreads();

    C[row * N + col] = sA[threadIdx.y][threadIdx.x] * sB[threadIdx.y][threadIdx.x];
}


void display_matrix(int n, float* A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i * n + j]<< ", ";
        }
        std::cout << std::endl;
    }
}

int zad1()
{
    const int N = 16;

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = std::rand() % 10;
        h_B[i] = std::rand() % 10;
        h_C[i] = 0;
    }

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(block_size, block_size);

    dim3 numBlocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    NaiveMM << <numBlocks, threadsperblock >> > (d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Zadanie 1: (naiwne z wykorzystaniem pamieci globalnej)" << std::endl;
    std::cout <<"h_A"<< std::endl;
    display_matrix(N, h_A);
    std::cout << "h_B" << std::endl;
    display_matrix(N, h_B);
    std::cout << "h_C" << std::endl;
    display_matrix(N, h_C);

    MM << <numBlocks, threadsperblock >> > (d_A, d_B, d_C, N);
    std::cout << "Zadanie 1: (naiwne z wykorzystaniem pamieci wspoldzielonej)" << std::endl;
    std::cout << "h_A" << std::endl;
    display_matrix(N, h_A);
    std::cout << "h_B" << std::endl;
    display_matrix(N, h_B);
    std::cout << "h_C" << std::endl;
    display_matrix(N, h_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}



__global__ void naiveNormalizeCols(const float* A, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float mean = 0.0f;
    for (int i = 0; i < N; i++)
        mean += A[i * N + col];
    mean /= N;

    float var = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float d = A[i * N + col] - mean;
        var += d * d;
    }
    float std = sqrtf(var / N);

    C[row * N + col] = (A[row * N + col] - mean) / std;
}
__global__ void normalizeColsShared(const float* A, float* C, int N)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;

    tile[threadIdx.y][threadIdx.x] = A[row * N + col];
    __syncthreads();

    float mean = 0.0f;
    for (int i = 0; i < N; i++)
        mean += A[i * N + col];
    mean /= N;

    float var = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float d = A[i * N + col] - mean;
        var += d * d;
    }
    float std = sqrtf(var / N);
    C[row * N + col] = (tile[threadIdx.y][threadIdx.x] - mean) / std;
}


int zad2()
{
    const int N = 4;

    float* h_A = new float[N * N];
    float* h_C = new float[N * N];

    float* d_A, * d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = std::rand() % 10;
        h_C[i] = 0;
    }

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(block_size, block_size);

    dim3 numBlocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    naiveNormalizeCols << <numBlocks, threadsperblock >> > (d_A, d_C, N);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Zadanie 2: (naiwne z wykorzystaniem pamieci globalnej)" << std::endl;
    std::cout << "h_A" << std::endl;
    display_matrix(N, h_A);
    std::cout << "h_C" << std::endl;
    display_matrix(N, h_C);

    normalizeColsShared << <numBlocks, threadsperblock >> > (d_A, d_C, N);
    std::cout << "Zadanie 2: (naiwne z wykorzystaniem pamieci wspoldzielonej)" << std::endl;
    std::cout << "h_A" << std::endl;
    display_matrix(N, h_A);
    std::cout << "h_C" << std::endl;
    display_matrix(N, h_C);

    delete[] h_A;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}

__global__ void tileSumNaive(const float* A, float* tileSums, int N)
{
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    int rowStart = tileRow * TILE_SIZE;
    int colStart = tileCol * TILE_SIZE;

    float sum = 0.0f;

    for (int r = 0; r < TILE_SIZE; r++)
    {
        for (int c = 0; c < TILE_SIZE; c++)
        {
            int row = rowStart + r;
            int col = colStart + c;

            if (row < N && col < N)
                sum += A[row * N + col];
        }
    }

    int tilesX = (N + TILE_SIZE - 1) / TILE_SIZE;
    tileSums[tileRow * tilesX + tileCol] = sum;
}

int zad3()
{
    const int N = 512;

    float* h_A = new float[N * N];
    float* h_C = new float[N * N  /32];

    float* d_A, * d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float) / 32);

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = std::rand() % 10;
    }
    for (int i = 0; i < N * N / 32; i++)
    {
        h_C[i] = 0;
    }

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(block_size, block_size);

    dim3 numBlocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    tileSumNaive << <numBlocks, threadsperblock >> > (d_A, d_C, N);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Zadanie 3: (naiwne z wykorzystaniem pamieci globalnej)" << std::endl;
    std::cout << "h_A" << std::endl;
    display_matrix(N, h_A);
    std::cout << "h_C" << std::endl;
    display_matrix(N / 32, h_C);

    //normalizeColsShared << <numBlocks, threadsperblock >> > (d_A, d_C, N);
    //std::cout << "Zadanie 2: (naiwne z wykorzystaniem pamieci wspoldzielonej)" << std::endl;
    //std::cout << "h_A" << std::endl;
    //display_matrix(N, h_A);
    //std::cout << "h_C" << std::endl;
    //display_matrix(N, h_C);

    delete[] h_A;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}

int main() {

    zad1();
    zad2();
    zad3();
    return 0;
}