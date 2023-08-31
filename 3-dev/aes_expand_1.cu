#include "util.h"
#include "sbox_E.h"
#include "aes_expand.h"

__global__
void aesExpand128_1_1(uint32_t *keyLeft, uint32_t *keyRight, OTblock *interleaved,
    OTblock *separated, OTblock *inData, uint64_t width) {

    uint32_t bx        = blockIdx.x;
    uint32_t tx        = threadIdx.x;
    uint32_t mod4tx = tx % 4;
    uint32_t tx4    = tx * 4;
    int x = 0;
    int expandDir = blockIdx.y;
    uint32_t *aesKey = expandDir == 0 ? keyLeft : keyRight;
    
    uint32_t stageBlockIdx[16] = {
        posIdx_E[mod4tx*4] + tx4,
        posIdx_E[mod4tx*4+1] + tx4,
        posIdx_E[mod4tx*4+2] + tx4,
        posIdx_E[mod4tx*4+3] + tx4,
        posIdx_E[mod4tx*4+4] + tx4,
        posIdx_E[mod4tx*4+5] + tx4,
        posIdx_E[mod4tx*4+6] + tx4,
        posIdx_E[mod4tx*4+7] + tx4,
        posIdx_E[mod4tx*4+8] + tx4,
        posIdx_E[mod4tx*4+9] + tx4,
        posIdx_E[mod4tx*4+10] + tx4,
        posIdx_E[mod4tx*4+11] + tx4,
        posIdx_E[mod4tx*4+12] + tx4,
        posIdx_E[mod4tx*4+13] + tx4,
        posIdx_E[mod4tx*4+14] + tx4,
        posIdx_E[mod4tx*4+15] + tx4,
    };

    uint64_t parentId = bx * AES_BSIZE + tx;
    uint64_t childId = 2 * parentId + expandDir;

    __shared__ UByte4 stageBlock1[4 * AES_BSIZE];
    __shared__ UByte4 stageBlock2[4 * AES_BSIZE];

    __shared__ UByte4 tBox0Block[4 * 256];
    __shared__ UByte4 tBox1Block[4 * 256];
    __shared__ UByte4 tBox2Block[4 * 256];
    __shared__ UByte4 tBox3Block[4 * 256];

    // input caricati in memoria
    OTblock inBlock = inData[parentId];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4+i].uival = inBlock.data[i];

    #pragma unroll
    for (int i = 0; i < 4; i++)
        tBox0Block[tx4+i].uival = TBox0[tx4+i];

    #pragma unroll
    for (int i = 0; i < 4; i++)
        tBox1Block[tx4+i].uival = TBox1[tx4+i];

    #pragma unroll
    for (int i = 0; i < 4; i++)
        tBox2Block[tx4+i].uival = TBox2[tx4+i];

    #pragma unroll
    for (int i = 0; i < 4; i++)
        tBox3Block[tx4+i].uival = TBox3[tx4+i];

    if (childId >= width) return;

    __syncthreads();

    //----------------------------------- 1st stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock2[tx4+i].uival = stageBlock1[tx4+i].uival ^ aesKey[i];

    __syncthreads();

    //-------------------------------- end of 1st stage --------------------------------


    //----------------------------------- 2nd stage -----------------------------------

    uint32_t op[16];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock2[stageBlockIdx[i]].ubval[i % 4];

    #pragma unroll
    for (int i = 0; i < 16; i+=4) {
        op[i] = tBox0Block[op[i]].uival;
        op[i+1] = tBox1Block[op[i+1]].uival;
        op[i+2] = tBox2Block[op[i+2]].uival;
        op[i+3] = tBox3Block[op[i+3]].uival;
    }
    
    x += 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4+i].uival = op[4*i]^op[4*i+1]^op[4*i+2]^op[4*i+3]^aesKey[x+i];

    __syncthreads();

    //-------------------------------- end of 2nd stage --------------------------------

    //----------------------------------- 3th stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock1[stageBlockIdx[i]].ubval[i % 4];

    #pragma unroll
    for (int i = 0; i < 16; i+=4) {
        op[i] = tBox0Block[op[i]].uival;
        op[i+1] = tBox1Block[op[i+1]].uival;
        op[i+2] = tBox2Block[op[i+2]].uival;
        op[i+3] = tBox3Block[op[i+3]].uival;
    }
    
    x += 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock2[tx4+i].uival = op[4*i]^op[4*i+1]^op[4*i+2]^op[4*i+3]^aesKey[x+i];

    __syncthreads();

    //-------------------------------- end of 3th stage --------------------------------

    //----------------------------------- 4th stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock2[stageBlockIdx[i]].ubval[i % 4];

    #pragma unroll
    for (int i = 0; i < 16; i+=4) {
        op[i] = tBox0Block[op[i]].uival;
        op[i+1] = tBox1Block[op[i+1]].uival;
        op[i+2] = tBox2Block[op[i+2]].uival;
        op[i+3] = tBox3Block[op[i+3]].uival;
    }
    
    x += 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4+i].uival = op[4*i]^op[4*i+1]^op[4*i+2]^op[4*i+3]^aesKey[x+i];

    __syncthreads();

    //-------------------------------- end of 4th stage --------------------------------

    //----------------------------------- 5th stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock1[stageBlockIdx[i]].ubval[i % 4];

    #pragma unroll
    for (int i = 0; i < 16; i+=4) {
        op[i] = tBox0Block[op[i]].uival;
        op[i+1] = tBox1Block[op[i+1]].uival;
        op[i+2] = tBox2Block[op[i+2]].uival;
        op[i+3] = tBox3Block[op[i+3]].uival;
    }
    
    x += 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock2[tx4+i].uival = op[4*i]^op[4*i+1]^op[4*i+2]^op[4*i+3]^aesKey[x+i];

    __syncthreads();

    //-------------------------------- end of 5th stage --------------------------------

    //----------------------------------- 6th stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock2[stageBlockIdx[i]].ubval[i % 4];

    #pragma unroll
    for (int i = 0; i < 16; i+=4) {
        op[i] = tBox0Block[op[i]].uival;
        op[i+1] = tBox1Block[op[i+1]].uival;
        op[i+2] = tBox2Block[op[i+2]].uival;
        op[i+3] = tBox3Block[op[i+3]].uival;
    }
    
    x += 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4+i].uival = op[4*i]^op[4*i+1]^op[4*i+2]^op[4*i+3]^aesKey[x+i];

    __syncthreads();

    //-------------------------------- end of 6th stage --------------------------------

    //----------------------------------- 7th stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock1[stageBlockIdx[i]].ubval[i % 4];

    #pragma unroll
    for (int i = 0; i < 16; i+=4) {
        op[i] = tBox0Block[op[i]].uival;
        op[i+1] = tBox1Block[op[i+1]].uival;
        op[i+2] = tBox2Block[op[i+2]].uival;
        op[i+3] = tBox3Block[op[i+3]].uival;
    }
    
    x += 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock2[tx4+i].uival = op[4*i]^op[4*i+1]^op[4*i+2]^op[4*i+3]^aesKey[x+i];

    __syncthreads();

    //-------------------------------- end of 7th stage --------------------------------

    //----------------------------------- 8th stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock2[stageBlockIdx[i]].ubval[i % 4];

    #pragma unroll
    for (int i = 0; i < 16; i+=4) {
        op[i] = tBox0Block[op[i]].uival;
        op[i+1] = tBox1Block[op[i+1]].uival;
        op[i+2] = tBox2Block[op[i+2]].uival;
        op[i+3] = tBox3Block[op[i+3]].uival;
    }
    
    x += 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4+i].uival = op[4*i]^op[4*i+1]^op[4*i+2]^op[4*i+3]^aesKey[x+i];

    __syncthreads();

    //-------------------------------- end of 8th stage --------------------------------

    //----------------------------------- 9th stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock1[stageBlockIdx[i]].ubval[i % 4];

    #pragma unroll
    for (int i = 0; i < 16; i+=4) {
        op[i] = tBox0Block[op[i]].uival;
        op[i+1] = tBox1Block[op[i+1]].uival;
        op[i+2] = tBox2Block[op[i+2]].uival;
        op[i+3] = tBox3Block[op[i+3]].uival;
    }
    
    x += 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock2[tx4+i].uival = op[4*i]^op[4*i+1]^op[4*i+2]^op[4*i+3]^aesKey[x+i];

    __syncthreads();

    //-------------------------------- end of 9th stage --------------------------------

    //----------------------------------- 10th stage -----------------------------------

    #pragma unroll
    for (int i = 0; i < 16; i++)
        op[i] = stageBlock2[stageBlockIdx[i]].ubval[i % 4];

    x += 4;

    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4].ubval[i] = tBox1Block[op[i]].ubval[3]^(aesKey[x]>>i*8 & 0x000000FF);
    
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4+1].ubval[i] = tBox1Block[op[i+4]].ubval[3]^(aesKey[x+1]>>i*8 & 0x000000FF);
    
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4+2].ubval[i] = tBox1Block[op[i+8]].ubval[3]^(aesKey[x+2]>>i*8 & 0x000000FF);
    
    #pragma unroll
    for (int i = 0; i < 4; i++)
        stageBlock1[tx4+3].ubval[i] = tBox1Block[op[i+12]].ubval[3]^(aesKey[x+3]>>i*8 & 0x000000FF);

    __syncthreads();

    //-------------------------------- end of 10th stage --------------------------------

    #pragma unroll
    for (int i = 0; i < 4; i++)
        interleaved[childId].data[i] = stageBlock1[tx4+i].uival;

    uint64_t offs = expandDir * (width / 2);
    #pragma unroll
    for (int i = 0; i < 4; i++)
        separated[parentId + offs].data[i] = stageBlock1[tx4+i].uival;
}