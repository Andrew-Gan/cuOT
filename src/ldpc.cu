#include "ldpc.h"

Matrix build_ldpc(int g, int t) {
    Matrix ldpc(32, 64);
    int colWeight = t - 1;

    for (int d = 0; d < DIAGONAL; d++) {
        int i = 0;
        int row = d * DIAGONAL_STRIDE + i, col = i;
        while(row < ROW && col < COLUMN / 2) {
            ldpc.set(row, col);
            i++;
        }
    }

    return ldpc;
}
