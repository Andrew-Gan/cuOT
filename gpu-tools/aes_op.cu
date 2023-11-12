#include <utility>
#include "aes_op.h"
#include "sbox_E.h"
#include "sbox_D.h"

union UByte4 {
  uint32_t uival;
  uint8_t ubval[4];
};

__device__
void swap(UByte4 *&a, UByte4 *&b) {
    UByte4 *tmp = a;
    a = b;
    b = tmp;
}

__global__
void aesEncrypt128(uint32_t *key, uint32_t * result, uint32_t * inData) {
	uint32_t bx		= blockIdx.x;
    uint32_t tx		= threadIdx.x;
    uint32_t mod4tx = tx%4;
    uint32_t int4tx = tx/4;
    uint32_t idx2	= int4tx*4;
	int x;

    uint32_t stageBlockIdx[4] = {
        posIdx_E[mod4tx*4]   + idx2,
        posIdx_E[mod4tx*4+1] + idx2,
        posIdx_E[mod4tx*4+2] + idx2,
        posIdx_E[mod4tx*4+3] + idx2,
    };

    __shared__ UByte4 stageBlock1[AES_BSIZE];
	__shared__ UByte4 stageBlock2[AES_BSIZE];

	__shared__ UByte4 tBox0Block[256];
	__shared__ UByte4 tBox1Block[256];
	__shared__ UByte4 tBox2Block[256];
	__shared__ UByte4 tBox3Block[256];

	// input caricati in memoria
	stageBlock1[tx].uival	= inData[AES_BSIZE * bx + tx ];

	uint32_t elemPerThread = 256/AES_BSIZE;
	for (uint32_t cnt=0; cnt<elemPerThread; cnt++) {
		tBox0Block[tx*elemPerThread + cnt].uival	= TBox0[tx*elemPerThread + cnt];
		tBox1Block[tx*elemPerThread + cnt].uival	= TBox1[tx*elemPerThread + cnt];
		tBox2Block[tx*elemPerThread + cnt].uival	= TBox2[tx*elemPerThread + cnt];
		tBox3Block[tx*elemPerThread + cnt].uival	= TBox3[tx*elemPerThread + cnt];
	}

	__syncthreads();

	//----------------------------------- 1st stage -----------------------------------

	x = mod4tx;
    stageBlock2[tx].uival = stageBlock1[tx].uival ^ key[x];

	__syncthreads();

	//-------------------------------- end of 1st stage --------------------------------

    UByte4 *sbSrc, *sbDes;
    uint32_t op[4];

    #pragma unroll
    for (int i = 1; i < 10; i++) {
        sbSrc = i % 2 == 0 ? stageBlock1 : stageBlock2;
        sbDes = i % 2 == 0 ? stageBlock2 : stageBlock1;
        
        op[0] = sbSrc[stageBlockIdx[0]].ubval[0];
        op[1] = sbSrc[stageBlockIdx[1]].ubval[1];
        op[2] = sbSrc[stageBlockIdx[2]].ubval[2];
        op[3] = sbSrc[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        sbDes[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

        swap(sbSrc, sbDes);
    }

	//----------------------------------- 11th stage -----------------------------------

    op[0] = sbSrc[stageBlockIdx[0]].ubval[0];
	op[1] = sbSrc[stageBlockIdx[1]].ubval[1];
	op[2] = sbSrc[stageBlockIdx[2]].ubval[2];
	op[3] = sbSrc[stageBlockIdx[3]].ubval[3];

	x += 4;

	sbDes[tx].ubval[3] = tBox1Block[op[3]].ubval[3]^( key[x]>>24);
	sbDes[tx].ubval[2] = tBox1Block[op[2]].ubval[3]^( (key[x]>>16) & 0x000000FF);
	sbDes[tx].ubval[1] = tBox1Block[op[1]].ubval[3]^( (key[x]>>8)  & 0x000000FF);
	sbDes[tx].ubval[0] = tBox1Block[op[0]].ubval[3]^( key[x]       & 0x000000FF);

	__syncthreads();

	//-------------------------------- end of 11th stage --------------------------------

	result[AES_BSIZE * bx + tx] = sbDes[tx].uival;
}

__global__
void aesDecrypt128(uint32_t *key, uint32_t * result, uint32_t * inData) {
	uint32_t bx		= blockIdx.x;
    uint32_t tx		= threadIdx.x;
    uint32_t mod4tx = tx%4;
    uint32_t int4tx = tx/4;
    uint32_t idx2	= int4tx*4;
	int x;

    uint32_t stageBlockIdx[4] = {
        posIdx_D[16 + mod4tx*4]   + idx2,
        posIdx_D[16 + mod4tx*4+1] + idx2,
        posIdx_D[16 + mod4tx*4+2] + idx2,
        posIdx_D[16 + mod4tx*4+3] + idx2
    };

    __shared__ UByte4 stageBlock1[AES_BSIZE];
    __shared__ UByte4 stageBlock2[AES_BSIZE];
    __shared__ UByte4 tBox0Block[AES_BSIZE];
    __shared__ UByte4 tBox1Block[AES_BSIZE];
    __shared__ UByte4 tBox2Block[AES_BSIZE];
    __shared__ UByte4 tBox3Block[AES_BSIZE];
    __shared__ UByte4 invSBoxBlock[AES_BSIZE];

	// input caricati in memoria
	stageBlock1[tx].uival	= inData[AES_BSIZE * bx + tx ];
	tBox0Block[tx].uival		= TBoxi0[tx];
	tBox1Block[tx].uival		= TBoxi1[tx];
	tBox2Block[tx].uival		= TBoxi2[tx];
	tBox3Block[tx].uival		= TBoxi3[tx];
	invSBoxBlock[tx].ubval[0]	= inv_SBox[tx];

	__syncthreads();

	//----------------------------------- 1st stage -----------------------------------

	x = mod4tx;
    stageBlock2[tx].uival = stageBlock1[tx].uival ^ key[x];

	__syncthreads();

	//-------------------------------- end of 1st stage --------------------------------

    UByte4 *sbSrc, *sbDes;
    uint32_t op[4];

    #pragma unroll
    for (int i = 1; i < 10; i++) {
        sbSrc = i % 2 == 0 ? stageBlock1 : stageBlock2;
        sbDes = i % 2 == 0 ? stageBlock2 : stageBlock1;
        
        op[0] = sbSrc[stageBlockIdx[0]].ubval[0];
        op[1] = sbSrc[stageBlockIdx[1]].ubval[1];
        op[2] = sbSrc[stageBlockIdx[2]].ubval[2];
        op[3] = sbSrc[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        sbDes[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

        swap(sbSrc, sbDes);
    }

	//----------------------------------- 11th stage -----------------------------------

    op[0] = sbSrc[stageBlockIdx[0]].ubval[0];
    op[1] = sbSrc[stageBlockIdx[1]].ubval[1];
    op[2] = sbSrc[stageBlockIdx[2]].ubval[2];
    op[3] = sbSrc[stageBlockIdx[3]].ubval[3];

	x += 4;

	sbDes[tx].ubval[3] = invSBoxBlock[op[3]].ubval[0]^( key[x]>>24);
	sbDes[tx].ubval[2] = invSBoxBlock[op[2]].ubval[0]^( key[x]>>16 & 0x000000FF);
	sbDes[tx].ubval[1] = invSBoxBlock[op[1]].ubval[0]^( key[x]>>8  & 0x000000FF);
	sbDes[tx].ubval[0] = invSBoxBlock[op[0]].ubval[0]^( key[x]     & 0x000000FF);

	__syncthreads();

	//-------------------------------- end of 11th stage --------------------------------

	result[AES_BSIZE * bx + tx] = sbDes[tx].uival;
}

__global__
void aesExpand128(uint32_t *keyLeft, uint32_t *keyRight, blk *interleaved,
    blk *separated, blk *inData, uint64_t width) {

    uint32_t bx        = blockIdx.x;
    uint32_t tx        = threadIdx.x;
    uint32_t mod4tx = tx % 4;
    uint32_t int4tx = tx / 4;
    uint32_t idx2    = int4tx * 4;
    int x;
    int expandDir = blockIdx.y;
    uint32_t *key = expandDir == 0 ? keyLeft : keyRight;

    uint32_t stageBlockIdx[4] = {
        posIdx_E[mod4tx*4] + idx2,
        posIdx_E[mod4tx*4+1] + idx2,
        posIdx_E[mod4tx*4+2] + idx2,
        posIdx_E[mod4tx*4+3] + idx2
    };

    uint64_t parentId =  (bx * AES_BSIZE + tx) / 4;
    uint64_t childId = 2 * parentId + expandDir;

    __shared__ UByte4 stageBlock1[AES_BSIZE];
    __shared__ UByte4 stageBlock2[AES_BSIZE];

    __shared__ UByte4 tBox0Block[256];
    __shared__ UByte4 tBox1Block[256];
    __shared__ UByte4 tBox2Block[256];
    __shared__ UByte4 tBox3Block[256];

    // input caricati in memoria
    stageBlock1[tx].uival    = inData[(blockDim.x * bx + tx) / 4].data[mod4tx];

    tBox0Block[tx].uival    = TBox0[tx];
    tBox1Block[tx].uival    = TBox1[tx];
    tBox2Block[tx].uival    = TBox2[tx];
    tBox3Block[tx].uival    = TBox3[tx];

    if (childId >= width) return;

    __syncthreads();

    //----------------------------------- 1st stage -----------------------------------

    x = mod4tx;
    stageBlock2[tx].uival = stageBlock1[tx].uival ^ key[x];

    __syncthreads();

    //-------------------------------- end of 1st stage --------------------------------

    UByte4 *sbSrc = stageBlock2, *sbDes = stageBlock1;
    uint32_t op[4];

    #pragma unroll
    for (int i = 1; i < 10; i++) {
        sbSrc = i % 2 == 0 ? stageBlock1 : stageBlock2;
        sbDes = i % 2 == 0 ? stageBlock2 : stageBlock1;
        
        op[0] = sbSrc[stageBlockIdx[0]].ubval[0];
        op[1] = sbSrc[stageBlockIdx[1]].ubval[1];
        op[2] = sbSrc[stageBlockIdx[2]].ubval[2];
        op[3] = sbSrc[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        sbDes[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

        swap(sbSrc, sbDes);
    }

    //----------------------------------- 11th stage -----------------------------------

    op[0] = sbSrc[stageBlockIdx[0]].ubval[0];
    op[1] = sbSrc[stageBlockIdx[1]].ubval[1];
    op[2] = sbSrc[stageBlockIdx[2]].ubval[2];
    op[3] = sbSrc[stageBlockIdx[3]].ubval[3];

    x += 4;
    sbDes[tx].ubval[3] = tBox1Block[op[3]].ubval[3]^( key[x]>>24);
    sbDes[tx].ubval[2] = tBox1Block[op[2]].ubval[3]^( (key[x]>>16) & 0x000000FF);
    sbDes[tx].ubval[1] = tBox1Block[op[1]].ubval[3]^( (key[x]>>8)  & 0x000000FF);
    sbDes[tx].ubval[0] = tBox1Block[op[0]].ubval[3]^( key[x]       & 0x000000FF);

    __syncthreads();

    //-------------------------------- end of 11th stage --------------------------------

    interleaved[childId].data[mod4tx] = sbDes[tx].uival;
    uint64_t offs = expandDir * (width / 2);
    separated[parentId + offs].data[mod4tx] = sbDes[tx].uival;
}