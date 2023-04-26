#include "util.h"

void mult_recver_cpu(Matrix ldpc, int *nonZeroRows, int numTrees);

void gemm_cpu(uint8_t *randomVec, Matrix ldpc, int *nonZeroRows, int numTrees, int start, int end);
