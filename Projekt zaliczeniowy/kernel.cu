#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Za³adowanie obrazu w skali szaroœci w postaci macierzy zawieraj¹cej wartoœci pikseli
// Implementacja kernela CUDA wykonuj¹cego operacjê splotu, gdzie ka¿dy watek oblicza jeden piksel obrazu wynikowego na podstawie kwadratowego okna filtra
// Ka¿dy kernel odczytuje dane z pamiêci globalnej
// brak koniecznoœci obs³ugi paddingu

__global__ void kernel(unsigned char* input, unsigned char* output, int width, int height, float* filter, int filterWidth) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = filterWidth / 2;

	if (x < offset || x >= (width - offset) || y < offset || y >= (height - offset)) {
		return;
	}

	float sum = 0.0f;

	for (int fy = -offset; fy <= offset; fy++) {
		for (int fx = -offset; fx <= offset; fx++) {
			int imgX = x + fx;
			int imgY = y + fy;

			int fltX = fx + offset;
			int fltY = fy + offset;

			int imgIdx = imgY * width + imgX;
			int fltIdx = fltY * filterWidth + fltX;

			sum += (float)input[imgIdx] * filter[fltIdx];
		}
	}

	if (sum < 0.0f) sum = 0.0f;
	if (sum > 255.0f) sum = 255.0f;

	int outIdx = y * width + x;
	output[outIdx] = (unsigned char)sum;
}

int main() {
	// ---Wczytanie obrazu z pliku---

	const char* filename = "obraz.jpg";
	int width, height, channels;

	unsigned char* h_image = stbi_load(filename, &width, &height, &channels, 1);

	if (h_image == nullptr) {
		return -1;
	}
	printf("\nWczytano obraz %dx%d", width, height);
	//printf("\nPrzykladowy piksel: %d", h_image[25]);

	// ---Zarezerwowanie i zape³nienie miejsca w pamieci karty graficznej
	int size = width * height * sizeof(unsigned char);
	unsigned char* obraz, * obraz_wynikowy;
	int filterWidth = 3;
	float h_filter[] = {
	0, -1, 0,
	-1, 5, -1,
	0, -1, 0
	};
	float* filter;
	int f_size = filterWidth * filterWidth * sizeof(float);
	
	cudaMalloc((void**)&obraz, size);
	cudaMalloc((void**)&obraz_wynikowy, size);
	cudaMalloc((void**)&filter, f_size);

	cudaMemcpy(obraz, h_image, size, cudaMemcpyHostToDevice);
	cudaMemcpy(filter, h_filter, f_size, cudaMemcpyHostToDevice);

	stbi_write_jpg("oryginalny_w_skali_szarosci.jpg", width, height, 1, h_image, 100);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	kernel << <gridSize, blockSize >> > (obraz, obraz_wynikowy, width, height, filter, filterWidth);

	cudaDeviceSynchronize();

	cudaMemcpy(h_image, obraz_wynikowy, size, cudaMemcpyDeviceToHost);

	int saveResult = stbi_write_jpg("wynik_splot.jpg", width, height, 1, h_image, 100);
	if (saveResult) {
		printf("\nZapisano plik pomyslnie!\n");
	}
	else {
		printf("Blad!\n");

	}

	// ---Zwolnienie pamieci
	stbi_image_free(h_image);
	cudaFree(obraz);
	cudaFree(obraz_wynikowy);
	cudaFree(filter);
	return 0;
}