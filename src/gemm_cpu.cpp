#include <thread>
#include <vector>
#include "gemm_cpu.h"

void gemm_cpu(uint8_t *randomVec, Matrix ldpc, int *nonZeroRows, int numTrees, int start, int end) {
    int col = 0, idx = 0;
    uint8_t res = 0, bit = 0;
    for (int row = start; row < end; row++) {
        for (int t = 0; t < (numTrees - 1) / 8 + 1; t++) {
            col = nonZeroRows[t];
            idx = row * ldpc.cols + col;
            bit = (ldpc.data[idx / 8] & (1 << idx % 8)) >> idx % 8;
            res ^= bit;
        }

        randomVec[row / 8] &= ~(1 << row % 8);
        randomVec[row / 8] |= res << row % 8;
    }
}

void mult_recver_cpu(Matrix ldpc, int *nonZeroRows, int numTrees) {
    int randVecSize = (ldpc.rows - 1) / 8 + 1;
    uint8_t *randomVec = (uint8_t*) malloc(randVecSize * sizeof(*randomVec));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    std::vector<std::thread> gemm;
    int numThread = ldpc.rows < 8 ? ldpc.rows : 8;
    int workload = ldpc.rows / numThread;
    for (int i = 0; i < numThread; i++) {
        int start = i * workload;
        int end = start + workload - 1;
        gemm.push_back(std::thread(gemm_cpu,randomVec, ldpc, nonZeroRows, numTrees, start, end));
    }
    for (int i = 0; i < numThread; i++) {
        gemm.at(i).join();
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    float duration = (end.tv_sec - start.tv_sec) * 1000;
    duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("Matrix mult using CPU: %0.4f ms\n", duration / NUM_SAMPLES);
}
