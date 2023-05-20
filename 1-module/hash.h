#include "util.h"
#include "gpu_block.h"

void hash_sender(Matrix rand, GPUBlock fullVec, int chunkC);
void hash_recver(Matrix rand, SparseVector choiceVec_d, Vector puncturedVec_d, int chunkR, int chunkC);
