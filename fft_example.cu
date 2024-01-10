#include "cufft.h"
#include <cstdio>

#define FFT_SIZE 4

__global__
void complex_mult(cufftComplex *a, cufftComplex *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float real = a[i].x * b[i].x - a[i].y * b[i].y;
        float im = a[i].x * b[i].y + a[i].y * b[i].x;
        a[i].x = real;
        a[i].y = im;
    }
}

__global__
void divider(cufftReal *data, int scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] /= scale;
}

int main() {
    cufftHandle plan[2];
    cufftCreate(&plan[0]);
    cufftCreate(&plan[1]);
    cufftPlan1d(&plan[0], 2 * FFT_SIZE, CUFFT_R2C, 1);
    cufftPlan1d(&plan[1], 2 * FFT_SIZE, CUFFT_C2R, 1);

    cufftReal *inH = new cufftReal[2 * FFT_SIZE];
    cufftComplex *midH = new cufftComplex[2 * FFT_SIZE];
    cufftReal *outH = new cufftReal[2 * FFT_SIZE];

    cufftReal *in;
    cudaMalloc(&in, 2 * FFT_SIZE * sizeof(cufftReal));
    cufftComplex *mid;
    cudaMalloc(&mid, 2 * FFT_SIZE * sizeof(cufftComplex));
    cufftReal *out;
    cudaMalloc(&out, 2 * FFT_SIZE * sizeof(cufftReal));

    for (int i = 0; i < FFT_SIZE; i++) {
        inH[i] = (cufftReal) i;
    }

    cudaMemcpy(in, inH, FFT_SIZE * sizeof(cufftReal), cudaMemcpyHostToDevice);
    printf("in:\n");
    for (int j = 0; j < FFT_SIZE; j++) {
        printf("%f ", inH[j]);
    }
    printf("\n");
    

    cufftExecR2C(plan[0], in, mid);

    complex_mult<<<1, 2 * FFT_SIZE>>>(mid, mid, 2 * FFT_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(midH, mid, 2 * FFT_SIZE * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    printf("mid:\n");
    for (int j = 0; j < 2 * FFT_SIZE; j++) {
        printf("%f + %f i\n", midH[j].x, midH[j].y);
    }

    cufftExecC2R(plan[1], mid, out);

    cudaMemcpy(outH, out, 2 * FFT_SIZE * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    printf("out:\n");
    for (int j = 0; j < 2 * FFT_SIZE; j++) {
        printf("%f\n", outH[j]);
    }

    divider<<<1, 2 * FFT_SIZE>>>(out, 2 * FFT_SIZE, 2 * FFT_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(outH, out, 2 * FFT_SIZE * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    printf("scaled and rounded:\n");
    for (int j = 0; j < 2 * FFT_SIZE; j++) {
        printf("%d ", (int) round(outH[j]));
    }
    printf("\n");

    delete[] inH;
    delete[] midH;
    delete[] outH;
    cudaFree(in);
    cudaFree(mid);
    cudaFree(out);
}
