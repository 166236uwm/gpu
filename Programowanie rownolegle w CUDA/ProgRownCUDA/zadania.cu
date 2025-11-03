#include "zadania.h"

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void zadanie1_kernel() {
  printf("\nThread index - %d", threadIdx.x);
}

void zadanie1() {
  // Przygotować kernel,
  // który zwraca w standardowym
  // wyjściu indeks każdego wątku.

  zadanie1_kernel<<<1, 64>>>();
  cudaDeviceSynchronize();
}

__global__ void zadanie2_kernel(int a) {
  printf("\nThread index %d multiplied by %d -> %d", threadIdx.x, a,
         threadIdx.x * a);
}

void zadanie2() {
  // Napisać program, w którym każdy wątek ma przypisaną zmienną
  // liczbową x (np. indeks warp’a). Każdy wątek powinien
  // pomnożyć swoją wartość przez wartość całkowitą (a)
  // przekazaną w parametrze do kernel i zwrócić wynik w wyjściu
  // standardowym.

  zadanie2_kernel<<<1, 32>>>(2);
  cudaDeviceSynchronize();
}

__global__ void zadanie3_kernel(int a) {
  if (threadIdx.x % a == 0)
    printf("\nThread index %d is divisible by %d", threadIdx.x, a);
}

void zadanie3() {
  // Napisać program, w którym każdy wątek sprawdza,
  // czy jego indeks wątku jest podzielny przez 3 i jeśli tak,
  // to zwraca odpowiedni komunikat.

  zadanie3_kernel<<<1, 64>>>(3);
  cudaDeviceSynchronize();
}

__global__ void zadanie4_kernel(int a) {
  printf("\nThread index %d, block index %d, num: %d", threadIdx.x, blockIdx.x,
         threadIdx.x + blockIdx.x + a);
}

void zadanie4() {
  // Przygotować program, w którym każdy wątek zaczyna z wartością
  // początkową 100 i dodaje do niej swój indeks wątku oraz indeks
  // bloku, a następnie zwraca wynik na wyjściu standardowym.

  zadanie4_kernel<<<3, 16>>>(100);
  cudaDeviceSynchronize();
}

__global__ void zadanie5_kernel() {
  int warpIdValue = 0;
  warpIdValue = threadIdx.x / 32;
  printf("\nSum of thread id %d, block id %d and warp id %d is ", threadIdx.x,
         blockIdx.x, warpIdValue);
  if ((threadIdx.x + blockIdx.x + warpIdValue) % 2 == 0) {
    printf("\nSum of thread id %d, block id %d and warp id %d is even",
           threadIdx.x, blockIdx.x, warpIdValue);
  } else {
    printf("\nSum of thread id %d, block id %d and warp id %d is not even",
           threadIdx.x, blockIdx.x, warpIdValue);
  }
}

void zadanie5() {
  // Przygotować program, w którym każdy wątek sprawdza,
  // czy suma indeksu wątku, bloku i warp’a jest parzysta i zwraca
  // odpowiedni komunikat.

  zadanie5_kernel<<<2, 16>>>();
  cudaDeviceSynchronize();
}

#define N 1024

__global__ void zadanie6_kernel(int *A, int *B, int *C, int n) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

void zadanie6() {
  // Przygotować program, który przyjmuje dwie tablice wejściowe 
  // A i B o rozmiarze N, a następnie w wektorze C
  // zapisze ich różnicę.

  // alokacja w pamięci RAM CPU
  int *A, *B, *C;
  int *dA, *dB, *dC;
  int size = N * sizeof(int);

  A = (int *)malloc(size);
  B = (int *)malloc(size);
  C = (int *)malloc(size);

  // alokacja w RAM GPU
  cudaMalloc((void **)&dA, size);
  cudaMalloc((void **)&dB, size);
  cudaMalloc((void **)&dC, size);

  // inicjalizacja wartości wektorów
  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = N - i;
  }

  // transfer wektorów z pamięci RAM CPU do RAM GPU
  cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

  // obliczenia
  zadanie6_kernel<<<1, N>>>(dA, dB, dC, size);
  cudaDeviceSynchronize();

  // transfer wyniku
  cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

  // wypisanie wyników
  for (int i = 0; i < N; i++) {
    printf("%d + %d = %d, ", A[i], B[i], C[i]);
  }

  // zwolnienie pamieci
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(A);
  free(B);
  free(C);
}

__global__ void zadanie7_kernel(int *A, int *B, int *C, int n) {
  int i = threadIdx.x;
  if (i % 2 == 0)
    C[i] = A[i] + B[i];
  else
    C[i] = A[i] - B[i];
}

void zadanie7() {
  // Przygotować program, który przyjmuje dwie tablice wejściowe 
  // A i B o rozmiarze N, a następnie w wektorze C zapisze ich sumę
  // dla elementów na pozycjach parzystych oraz różnicę dla
  // elementów na pozycjach nieparzystych.

  // alokacja w pamięci RAM CPU
  int *A, *B, *C;
  int *dA, *dB, *dC;
  int size = N * sizeof(int);

  A = (int *)malloc(size);
  B = (int *)malloc(size);
  C = (int *)malloc(size);

  // alokacja w RAM GPU
  cudaMalloc((void **)&dA, size);
  cudaMalloc((void **)&dB, size);
  cudaMalloc((void **)&dC, size);

  // inicjalizacja wartości wektorów
  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = N - i;
  }

  // transfer wektorów z pamięci RAM CPU do RAM GPU
  cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

  // obliczenia
  zadanie7_kernel<<<1, N>>>(dA, dB, dC, size);
  cudaDeviceSynchronize();

  // transfer wyniku
  cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

  // wypisanie wyników
  for (int i = 0; i < N; i++) {
    printf("%d ", A[i]);
    if (i % 2 == 0)
      printf("+");
    else
      printf("-");
    printf(" %d = %d, ", B[i], C[i]);
  }

  // zwolnienie pamieci
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(A);
  free(B);
  free(C);
}

__global__ void zadanie8_kernel(int *A, int *B, int *C, int n) {
  int i = threadIdx.x;
  int warpIdValue;
  warpIdValue = threadIdx.x / 32;
  if (warpIdValue % 2 == 0)
    C[i] = A[i] + B[i];
  else
    C[i] = A[i] - B[i];
}

void zadanie8() {
  // Przygotować program, który przyjmuje dwie tablice wejściowe 
  // A i B o rozmiarze N, a następnie w wektorze C zapisze ich sumę
  // dla elementów w warp’ach na pozycjach parzystych oraz różnicę
  // dla elementów w warp’ach na pozycjach nieparzystych.

  // alokacja w pamięci RAM CPU
  int *A, *B, *C;
  int *dA, *dB, *dC;
  int size = N * sizeof(int);

  A = (int *)malloc(size);
  B = (int *)malloc(size);
  C = (int *)malloc(size);

  // alokacja w RAM GPU
  cudaMalloc((void **)&dA, size);
  cudaMalloc((void **)&dB, size);
  cudaMalloc((void **)&dC, size);

  // inicjalizacja wartości wektorów
  for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = N - i;
  }

  // transfer wektorów z pamięci RAM CPU do RAM GPU
  cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

  // obliczenia
  zadanie8_kernel<<<1, N>>>(dA, dB, dC, size);
  cudaDeviceSynchronize();

  // transfer wyniku
  cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

  // wypisanie wyników
  for (int i = 0; i < N; i++) {
    printf("%d, ", C[i]);
  }

  // zwolnienie pamieci
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(A);
  free(B);
  free(C);
}

__global__ void zadanie9_kernel(int *A, int *B, int *C, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void zadanie9() {
  // Przygotować program, który wykona sumę dwóch bardzo dużych
  // wektorów, łącznie przekraczających rozmiar pamięci operacyjnej.
  // W tym celu należy wykonać następujące kroki:

  // wektory nie powinny być wczytywane w całości,
  // a jedynie w formie części o ustalonej długości;
  // jeżeli długość wektora nie pozwala na ustalenie
  // części o równej długości ostatnia może być krótsza,

  // sumowaniu równoległemu podlega każda para części z osobna
  // będące odpowiednikami wektorów A i B

  // suma części będzie przechowywana w wektorze C dla każdej
  // pary części wektorów,

  // każda wersja wektora C będzie przechowywana w zewnętrznym
  // źródle układając ostateczny wektor z części.

  // alokacja w pamięci RAM CPU
  int *A, *B, *C;
  int *dA, *dB, *dC;
  int size = N * sizeof(int);
  int opMem = 128;

  A = (int *)malloc(size);
  B = (int *)malloc(size);
  C = (int *)malloc(size);

  // inicjalizacja wartości wektorów
  for (int i = 0; i < N; i++) {
    A[i] = 1;
    B[i] = 3;
  }

  // alokacja w RAM GPU
  // symulujemy przekroczenie pamięci operacyjnej
  // dla przykładu pamięć operacyjna to 512
  if (size > opMem * sizeof(int)) {
    int batchCount = (N + opMem - 1) / opMem;
    int batchMemSize = opMem * sizeof(int);
    cudaMalloc((void **)&dA, batchMemSize);
    cudaMalloc((void **)&dB, batchMemSize);
    cudaMalloc((void **)&dC, batchMemSize);

    for (int i = 0; i < batchCount; i++) {
      int currentBatchElements;
      if (i == batchCount - 1) {
        currentBatchElements = N - i * opMem;
      } else {
        currentBatchElements = opMem;
      }
      int currentBatchSize = currentBatchElements * sizeof(int);
      int offset = i * opMem;

      cudaMemcpy(dA, A + offset, currentBatchSize, cudaMemcpyHostToDevice);
      cudaMemcpy(dB, B + offset, currentBatchSize, cudaMemcpyHostToDevice);

      int threadsPerBlock = 256;
      int blocks =
          (currentBatchElements + threadsPerBlock - 1) / threadsPerBlock;

      zadanie9_kernel<<<blocks, threadsPerBlock>>>(dA, dB, dC,
                                                   currentBatchElements);
      cudaDeviceSynchronize();
      cudaMemcpy(C + offset, dC, currentBatchSize, cudaMemcpyDeviceToHost);
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  } else {
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dC, size);

    // transfer wektorów z pamięci RAM CPU do RAM GPU
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    // obliczenia
    zadanie9_kernel<<<1, N>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    // transfer wyniku
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    // zwolnienie pamieci
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
  for (int i = 0; i < N; i++) {
    printf("C[%d] = %d, ", i, C[i]);
  }
  free(A);
  free(B);
  free(C);
}