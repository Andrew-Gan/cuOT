#include "mytypes.h"

void mult_sender_gpu(Matrix rand, Vector d_fullVec, int chunkC);
void mult_recver_gpu(Matrix rand, Vector d_choiceVec, Vector d_puncturedVec, int chunkC);
