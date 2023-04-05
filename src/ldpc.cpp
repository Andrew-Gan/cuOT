#include "ldpc.h"

void print_matrix(Matrix& mat) {
    for(int r = 0; r < mat.rows; r++) {
        for(int c = 0; c < mat.cols; c++) {
            int idx = r * mat.cols + c;
            int bit = mat.data[idx / 8];
            bit &= 1 << (idx % 8);
            bit >>= idx % 8;
            printf("%d", bit);
        }
        printf("\n");
    }
}

Matrix generate_ldpc(int numLeaves, int numTrees) {
    Matrix ldpc;
    ldpc.rows = numLeaves / 2;
    ldpc.cols = numLeaves;
    ldpc.data = (uint8_t*) calloc(ldpc.rows * ldpc.cols, 1);

    // generate left
    int gap = ldpc.rows / numTrees;
    for (int i = 0; i < numTrees; i++) {
        int row = i * gap;
        int col = 0;
        while (row < ldpc.rows && col < ldpc.cols / 2) {
            int idx = row++ * ldpc.cols + col++;
            ldpc.data[idx / 8] |= 1 << (idx % 8);
        }
    }
    for (int i = 1; i < numTrees; i++) {
        int r = 0;
        int c = i * gap;
        while (r < ldpc.rows && c < ldpc.cols / 2) {
            int idx = r++ * ldpc.cols + c++;
            ldpc.data[idx / 8] |= 1 << (idx % 8);
        }
    }

    // generate right
    uint8_t sampled_rows[4] = {0x42, 0x49, 0xea, 0xde};
    for (int r = 0; r < ldpc.rows; r++) {
        int c = ldpc.cols / 2 + r;
        int idx = r * ldpc.cols + c;
        ldpc.data[idx / 8] = sampled_rows[idx % 4];
    }

    return ldpc;
}
