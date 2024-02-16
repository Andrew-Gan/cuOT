#include "cufft.h"
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cassert>

#define BATCH_SIZE 1
#define FFT_SIZE (1<<4)
#define SAMPLE_SIZE 1

__global__
void complex_mult(cufftComplex *a, cufftComplex *b, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    int i = row * n + col;
    if (col < n) {
        float real = a[i].x * b[i].x - a[i].y * b[i].y;
        float im = a[i].x * b[i].y + a[i].y * b[i].x;
        a[i].x = real;
        a[i].y = im;
    }
}

__global__
void divider(cufftReal *data, int scale, int n) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * n + col;
    if (col < n) data[i] /= scale;
}

int main() {
    cudaSetDevice(0);
    cufftHandle aPlan = 0, bPlan = 0;
    cufftPlan1d(&aPlan, FFT_SIZE, CUFFT_R2C, BATCH_SIZE);
    cufftPlan1d(&bPlan, FFT_SIZE, CUFFT_C2R, BATCH_SIZE);

    cufftReal inH[FFT_SIZE], outH[FFT_SIZE];
    // cufftComplex midH[FFT_SIZE / 2 + 1];
    memset(inH, 0, sizeof(inH));

    cufftReal *in, *out;
    cufftComplex *mid;
    cudaMalloc(&in, BATCH_SIZE * FFT_SIZE * sizeof(cufftReal));
    cudaMalloc(&out, BATCH_SIZE * FFT_SIZE * sizeof(cufftReal));
    cudaMalloc(&mid, BATCH_SIZE * (FFT_SIZE / 2 + 1) * sizeof(cufftComplex));

    for (int i = 0; i < FFT_SIZE / 2; i++) {
        inH[i] = 1.0f;
    }
    // printf("in:\n");
    // for (int j = 0; j < FFT_SIZE; j++)
    //     printf("%.2f ", inH[j]);

    cudaMemcpy(in, inH, sizeof(inH), cudaMemcpyHostToDevice);
    
    struct timespec tp[2];

    clock_gettime(CLOCK_MONOTONIC, &tp[0]);

    uint64_t nThread, block;
    dim3 grid;

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        cufftExecR2C(aPlan, in, mid);
        // cudaMemcpy(midH, mid, sizeof(midH), cudaMemcpyDeviceToHost);
        // printf("mid:\n");
        // for (int j = 0; j < FFT_SIZE / 2 + 1; j++)
        //     printf("%.4f + %.4f i\n", midH[j].x, midH[j].y);

        nThread = FFT_SIZE / 2 + 1;
        block = std::min(1024UL, nThread);
        grid = dim3((nThread + block - 1) / block, BATCH_SIZE);

        complex_mult<<<grid, block>>>(mid, mid, nThread);
        cudaDeviceSynchronize();

        // cudaMemcpy(midH, mid, sizeof(mid), cudaMemcpyDeviceToHost);
        // printf("mid:\n");
        // for (int j = 0; j < FFT_SIZE; j++)
        //     printf("%.4f + %.4f i\n", midH[j].x, midH[j].y);

        cufftExecC2R(bPlan, mid, out);

        // cudaMemcpy(outH, out, sizeof(outH), cudaMemcpyDeviceToHost);
        // printf("out:\n");
        // for (int j = 0; j < FFT_SIZE; j++)
        //     printf("%f\n", outH[j]);

        nThread = FFT_SIZE;
        block = std::min(1024UL, nThread);
        grid = dim3((nThread + block - 1) / block, BATCH_SIZE);

        // divider<<<grid, block>>>(out, FFT_SIZE, FFT_SIZE);
        cudaDeviceSynchronize();
        cudaMemcpy(outH, out, sizeof(outH), cudaMemcpyDeviceToHost);
        printf("scaled and rounded:\n");
        for (int j = 0; j < FFT_SIZE; j++)
            printf("%.2f ", outH[j]);
    }

    clock_gettime(CLOCK_MONOTONIC, &tp[1]);

    float duration = (float)(tp[1].tv_sec-tp[0].tv_sec) * 1000;
    duration += (float)(tp[1].tv_nsec-tp[0].tv_nsec) / 1000000;
    printf("\nFFT duration: %.2f ms\n", duration / SAMPLE_SIZE);

    cudaFree(in);
    cudaFree(mid);
    cudaFree(out);
}
