#include "mytypes.h"

void mult_recver_gpu(Matrix ldpc, TreeNode *d_multiPprf, int *nonZeroRows, int numTrees);

__global__
void gemm_gpu(uint8_t *randomVec, Matrix *ldpc, int *nonZeroRows, int numTrees);
