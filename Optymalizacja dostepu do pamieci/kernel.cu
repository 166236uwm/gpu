
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
	__shared__ int s_dC[1024];
	int i = threadIdx.x + 256 * blockIdx.x;
	s_dC[i] = dA[i] + dB[i];
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

	printf("Wywoluej kernele...\n");
	SumOfTwoVectors <<<blocks, threadsPerBlock >>> (dA, dB, dC);
	printf("wywolano kernele\n");
	cudaDeviceSynchronize();
	printf("zsynchronizowano kernele\n");
	cudaMemcpy(A, dA, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(B, dB, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);
	printf("skopoiowano zawartosc pamieci GPU\n");
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


__global__ void warpShuffle(int* dA, int* dB) {
	int idx = threadIdx.x;
	int val = dA[idx];

	unsigned mask = 0xffffffff;
	int rev = __shfl_sync(mask, val, 31 - idx);

	dB[idx] = rev;
}

void zadanie3() {
	const int n = 32;
	int* A, * B;
	int* dA, * dB;
	int size = n * sizeof(int);

	A = (int*)malloc(size);
	B = (int*)malloc(size);

	cudaMalloc((void**)&dA, size);
	cudaMalloc((void**)&dB, size);

	for (int i = 0; i < n; i++)
	{
		A[i] = i;
	}

	cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);


	printf("Wywoluej kernele...\n");
	warpShuffle << <1, 32 >> > (dA, dB);
	printf("wywolano kernele\n");
	cudaDeviceSynchronize();
	printf("zsynchronizowano kernele\n");
	cudaMemcpy(B, dB, size, cudaMemcpyDeviceToHost);
	printf("skopoiowano zawartosc pamieci GPU\n");
	for (int i = 0; i < n; i++) {
		printf("A[%d] (%d) zamieniono na B[%d] (%d)\n", i, A[i], i, B[i]);
	}

	cudaFree(dA);
	cudaFree(dB);
	free(A);
	free(B);
}


__global__ void ReduceWithWarpOptimization(float* input, int n) {
	// pamiec wspoldzielona do trzymania sum z warp'ow, rozmiar elastyczny
	extern __shared__ float shared[];
	int tid = threadIdx.x;
	int index = 2 * blockIdx.x * blockDim.x + tid;
	float sum = 0; // wartosc do przechowywania w rejestrze
	sum = (index < n ? input[index] : 0.0f) + (index + blockDim.x < n ? input[index + blockDim.x] : 0.0f);

	// petla sumujaca wartosci z rejestrow tego warpu
		// offset to kolejno 16, 8, ..., 1
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
		// sum = obecna sum z tego watku + wartosc watku od ID 
		// offset w tym warp'ie
		sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
	}

	// jezeli biezacy id watku to pierwszy w warp'ie
	if (tid % warpSize == 0) {
		// ustaw wartosc sumy tego warp'u w pamieci wspoldzielonej 
		// na jego pozycji
		shared[tid / warpSize] = sum;
	}
	__syncthreads();

	// jezeli id obecnego watku miesci sie w warpie, to uzyj go do wykonania akcji
	if (tid < warpSize) {
		// jezeli id watku odpowiada pozycji wartosci w tablicy shared
		sum = (tid < (blockDim.x / warpSize)) ? shared[tid] : 0.0f;
		// offset po kolei 16, 8, ..., 1
		for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
			// zredukuj sumy w warpie, zawarte w rejestrach, 
			// to sa sumy czastkowe ze wszystkich warp'ow z bloku
			sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
		}
	}

	if (tid == 0) { // jezeli pierwszy watek w bloku
		input[blockIdx.x] = sum; // ustaw sume wszystkich warp'ow
	}
}

int cpuReduce(int* input, int size) {
	int sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += input[i];
	}
	return sum;
}

void zadanie4() {
	int n = 1024 * 1024;
	size_t bytes = n * sizeof(float);

	float* h_input = new float[n];
	float* d_input;

	for (int i = 0; i < n; i++) {
		h_input[i] = i * 1.0;
	}

	cudaMalloc(&d_input, bytes);

	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

	int block_size = 256;
	int grid_size = (n + 2 * block_size - 1) / (2 * block_size);
	size_t shared_mem_size = (block_size / 32) * sizeof(float);

	while (grid_size > 1) {
		ReduceWithWarpOptimization << <grid_size, block_size, shared_mem_size >> > (d_input, n);
		cudaDeviceSynchronize();

		n = grid_size;
		grid_size = (n + 2 * block_size - 1) / (2 * block_size);
	}

	ReduceWithWarpOptimization << <1, block_size, shared_mem_size >> > (d_input, n);
	cudaDeviceSynchronize();

	cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost);

	printf("\nSum: %f \nMean: %f, \n", h_input[0], h_input[0] / n);

	cudaFree(d_input);
	delete[] h_input;

}

int main() {
	printf("\n------------------------------\nWykonuje zadanie 1\n------------------------------\n");
	zadanie1();
	printf("\n------------------------------\nWykonuje zadanie 2\n------------------------------\n");
	zadanie2();
	printf("\n------------------------------\nWykonuje zadanie 3\n------------------------------\n");
	zadanie3();
	printf("\n------------------------------\nWykonuje zadanie 4\n------------------------------\n");
	zadanie4();

	return 0;
}