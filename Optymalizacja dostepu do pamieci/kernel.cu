
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace std;

// Przygotuj kernel, który mno¿y kazdy element wektora przez sta³a wartoœæ skalarn¹. 
// Wykorzystaæ w tym celu pamiêæ wspó³dzielon¹, aby przyspierzyæ odczyt danych

__global__ void VecTimesScalar(int* dvec, int* dres, int skal) {
	extern __shared__ int skalar;
	skalar = skal;
	__syncthreads();
	int i = threadIdx.x + 256 * blockIdx.x;
	dres[i] = dvec[i] * skalar;

}

void zadanie1() {
	const int n = 1024;
	int skalar = 7;
	int* vec, * res;
	int* dvec, * dres;
	int s = n * sizeof(int);
	vec = (int*)malloc(s);
	res = (int*)malloc(s);

	cudaMalloc((void**)&dvec, s);
	cudaMalloc((void**)&dres, s);

	for (int i = 0; i < n; i++) {
		vec[i] = 17 + i % 18;
	}

	cudaMemcpy(dvec, vec, s, cudaMemcpyHostToDevice);
	int threadsPerBlock = 256;
	int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
	VecTimesScalar << <blocks, threadsPerBlock >> > (dvec, dres, skalar);

	cudaMemcpy(res, dres, s, cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++) {
		printf("%d --> %d * %d = %d\n", i, vec[i], skalar, res[i]);
	}
	cudaFree(dvec);
	cudaFree(dres);
	free(vec);
	free(res);
}

// Przygotowaæ kernel, który sumuje dwa wektory element po elemencie. 
// Wykorzystaæ w tym celu pamiêæ wspó³dzielon¹, 
// aby zminimalizowaæ liczbê odwo³añ do pamiêci globalnej.

__global__ void SumOfTwoVectors(int* dA, int* dB, int* dC){
	extern __shared__ int s_dC[];
	int i = threadIdx.x + 256 * blockIdx.x;
	s_dC[i] = dA[i] + dB[i];
	__syncthreads();
	dC[i] = s_dC[i];
	__syncthreads();
}

void zadanie2() {
	const int n = 1024;
	int* A, * B, * C;
	int* dA, * dB, * dC;
	int size = n * sizeof(int);

	A = (int*)malloc(size);
	B = (int*)malloc(size);
	C = (int*)malloc(size);

	cudaMalloc((void**)&dA, size);
	cudaMalloc((void**)&dB, size);
	cudaMalloc((void**)&dC, size);

	for (int i = 0; i < n; i++)
	{
		A[i] = i * 5;
		B[i] = (n - i) - 67;
	}

	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
	SumOfTwoVectors <<<blocks, threadsPerBlock >>> (dA, dB, dC);

	cudaMemcpy(A, dA, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(B, dB, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++) {
		printf("%d --> %d + %d = %d\n", i, A[i], B[i], C[i]);
	}

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	free(A);
	free(B);
	free(C);
}

int main() {
	printf("\n------------------------------\nWykonuje zadanie 1\n------------------------------\n");
	zadanie1();
	printf("\n------------------------------\nWykonuje zadanie 2\n------------------------------\n");
	zadanie2();

	return 0;
}