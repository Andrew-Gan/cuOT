#include "util.h"
#include "gpu_block.h"

#define CHUNK_SIDE (1<<18)

void hash_sender(GPUBlock &fullVectorHashed, Matrix &randMatrix, GPUBlock &fullVector, int chunkC);
void hash_recver(GPUBlock &puncVectorHashed, GPUBlock &choiceVectorHashed, Matrix &randMatrix, GPUBlock &puncVector, SparseVector &choiceVector, int chunkR, int chunkC);
