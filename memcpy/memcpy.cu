#include <iostream>

#define SAMPLE_SIZE 8

int main() {
    unsigned char *host0, *host1, *dev0a, *dev0b, *dev1;
    const size_t min = 15, max = 25;

    host0 = new unsigned char[1 << max];
    host1 = new unsigned char[1 << max];
    cudaSetDevice(0);
    cudaMalloc(&dev0a, 1 << max);
    cudaMalloc(&dev0b, 1 << max);
    cudaSetDevice(1);
    cudaMalloc(&dev1, 1 << max);

    cudaSetDevice(0);
    struct timespec t[2];

    std::cout << "h2h:";
    for (size_t size = min; size <= max; size++) {
        clock_gettime(CLOCK_MONOTONIC, &t[0]);
        for (int i = 0; i < SAMPLE_SIZE; i++)
            cudaMemcpy(host1, host0, 1 << size, cudaMemcpyHostToHost);
        clock_gettime(CLOCK_MONOTONIC, &t[1]);
        float duration = (float)(t[1].tv_sec - t[0].tv_sec) * 1000;
        duration += (float)(t[1].tv_nsec - t[0].tv_nsec) / 1000000;
        std::cout << duration / SAMPLE_SIZE << ",";
    }
    std::cout << std::endl;

    std::cout << "h2d:";
    for (size_t size = min; size <= max; size++) {
        clock_gettime(CLOCK_MONOTONIC, &t[0]);
        for (int i = 0; i < SAMPLE_SIZE; i++)
            cudaMemcpy(dev0a, host0, 1 << size, cudaMemcpyHostToDevice);
        clock_gettime(CLOCK_MONOTONIC, &t[1]);
        float duration = (float)(t[1].tv_sec - t[0].tv_sec) * 1000;
        duration += (float)(t[1].tv_nsec - t[0].tv_nsec) / 1000000;
        std::cout << duration / SAMPLE_SIZE << ",";
    }
    std::cout << std::endl;

    std::cout << "d2h:";
    for (size_t size = min; size <= max; size++) {
        clock_gettime(CLOCK_MONOTONIC, &t[0]);
        for (int i = 0; i < SAMPLE_SIZE; i++)
            cudaMemcpy(host0, dev0a, 1 << size, cudaMemcpyDeviceToHost);
        clock_gettime(CLOCK_MONOTONIC, &t[1]);
        float duration = (float)(t[1].tv_sec - t[0].tv_sec) * 1000;
        duration += (float)(t[1].tv_nsec - t[0].tv_nsec) / 1000000;
        std::cout << duration / SAMPLE_SIZE << ",";
    }
    std::cout << std::endl;

    std::cout << "d2d:";
    for (size_t size = min; size <= max; size++) {
        clock_gettime(CLOCK_MONOTONIC, &t[0]);
        for (int i = 0; i < SAMPLE_SIZE; i++)
            cudaMemcpy(dev0b, dev0a, 1 << size, cudaMemcpyDeviceToDevice);
        clock_gettime(CLOCK_MONOTONIC, &t[1]);
        float duration = (float)(t[1].tv_sec - t[0].tv_sec) * 1000;
        duration += (float)(t[1].tv_nsec - t[0].tv_nsec) / 1000000;
        std::cout << duration / SAMPLE_SIZE << ",";
    }
    std::cout << std::endl;

    std::cout << "peer:";
    for (size_t size = min; size <= max; size++) {
        clock_gettime(CLOCK_MONOTONIC, &t[0]);
        for (int i = 0; i < SAMPLE_SIZE; i++)
            cudaMemcpyPeer(dev1, 1, dev0a, 0, 1 << size);
        clock_gettime(CLOCK_MONOTONIC, &t[1]);
        float duration = (float)(t[1].tv_sec - t[0].tv_sec) * 1000;
        duration += (float)(t[1].tv_nsec - t[0].tv_nsec) / 1000000;
        std::cout << duration / SAMPLE_SIZE << ",";
    }
    std::cout << std::endl;

    delete[] host0;
    delete [] host1;
    cudaSetDevice(0);
    cudaFree(dev0a);
    cudaFree(dev0b);
    cudaSetDevice(1);
    cudaFree(dev1);

    return EXIT_SUCCESS;
}
