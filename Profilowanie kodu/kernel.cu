
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

using namespace std;

#define N (1024 * 1024) 

__global__ void vectorAdd_kernel(int* A, int* B, int* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void zadanie_profilowanie(int numBlocks, int threadsPerBlock)
{
    int* A, * B, * C;
    int* dA, * dB, * dC;
    int size = N * sizeof(int);

    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    cudaMalloc((void**)&dA, size);
    cudaMalloc((void**)&dB, size);
    cudaMalloc((void**)&dC, size);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 1;
    }

    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    printf("Uruchamiam kernel z konfiguracja: Bloki=%d, Watki=%d\n", numBlocks, threadsPerBlock);
    vectorAdd_kernel <<<numBlocks, threadsPerBlock >>> (dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    printf("Pierwszy element: %d + %d = %d\n", A[0], B[0], C[0]);
    printf("Ostatni: %d + %d = %d\n", A[N - 1], B[N - 1], C[N - 1]);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);
}

int main()
{
	// threadsPerBlock -> min 32 -> max 1024
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nStart profilowania...\n");

    zadanie_profilowanie(numBlocks, threadsPerBlock);

    printf("\nProfilowanie zakonczone.\n");
    return 0;
}