#include <utility>
#include "aes_op.h"
#include "sbox_E.h"
#include "sbox_D.h"

union UByte4 {
  uint32_t uival;
  uint8_t ubval[4];
};

__global__
void aesEncrypt128(uint32_t *key, uint32_t * aesData) {
    uint32_t bx		= blockIdx.x;
    uint32_t tx		= threadIdx.x;
    uint32_t mod4tx = tx%4;
    uint32_t int4tx = tx/4;
    uint32_t idx2	= int4tx*4;
	int x;
    uint32_t y      = blockIdx.y * blockDim.y + threadIdx.y;
    key += y * 11 * AES_KEYLEN / sizeof(*key);

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
	stageBlock1[tx].uival	= aesData[AES_BSIZE * bx + tx ];

    tBox0Block[tx].uival	= TBox0[tx];
    tBox1Block[tx].uival	= TBox1[tx];
    tBox2Block[tx].uival	= TBox2[tx];
    tBox3Block[tx].uival	= TBox3[tx];

	//----------------------------------- 1st stage -----------------------------------

	x = mod4tx;
    stageBlock2[tx].uival = stageBlock1[tx].uival ^ key[x];

	//-------------------------------- end of 1st stage --------------------------------

    uint32_t op[4];

    #pragma unroll
    for (int i = 1; i < 9; i+=2) {
        op[0] = stageBlock2[stageBlockIdx[0]].ubval[0];
        op[1] = stageBlock2[stageBlockIdx[1]].ubval[1];
        op[2] = stageBlock2[stageBlockIdx[2]].ubval[2];
        op[3] = stageBlock2[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        stageBlock1[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

        op[0] = stageBlock1[stageBlockIdx[0]].ubval[0];
        op[1] = stageBlock1[stageBlockIdx[1]].ubval[1];
        op[2] = stageBlock1[stageBlockIdx[2]].ubval[2];
        op[3] = stageBlock1[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        stageBlock2[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];
    }

    op[0] = stageBlock2[stageBlockIdx[0]].ubval[0];
    op[1] = stageBlock2[stageBlockIdx[1]].ubval[1];
    op[2] = stageBlock2[stageBlockIdx[2]].ubval[2];
    op[3] = stageBlock2[stageBlockIdx[3]].ubval[3];

    op[0] = tBox0Block[op[0]].uival;
    op[1] = tBox1Block[op[1]].uival;
    op[2] = tBox2Block[op[2]].uival;
    op[3] = tBox3Block[op[3]].uival;

    x += 4;
    stageBlock1[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

	//----------------------------------- 11th stage -----------------------------------

    op[0] = stageBlock1[stageBlockIdx[0]].ubval[0];
	op[1] = stageBlock1[stageBlockIdx[1]].ubval[1];
	op[2] = stageBlock1[stageBlockIdx[2]].ubval[2];
	op[3] = stageBlock1[stageBlockIdx[3]].ubval[3];

	x += 4;
	stageBlock2[tx].ubval[3] = tBox1Block[op[3]].ubval[3]^( key[x]>>24);
	stageBlock2[tx].ubval[2] = tBox1Block[op[2]].ubval[3]^( (key[x]>>16) & 0x000000FF);
	stageBlock2[tx].ubval[1] = tBox1Block[op[1]].ubval[3]^( (key[x]>>8)  & 0x000000FF);
	stageBlock2[tx].ubval[0] = tBox1Block[op[0]].ubval[3]^( key[x]       & 0x000000FF);

	//-------------------------------- end of 11th stage --------------------------------

	aesData[AES_BSIZE * bx + tx] = stageBlock2[tx].uival;
}

__global__
void aesDecrypt128(uint32_t *key, uint32_t *aesData) {
	uint32_t bx		= blockIdx.x;
    uint32_t tx		= threadIdx.x;
    uint32_t mod4tx = tx%4;
    uint32_t int4tx = tx/4;
    uint32_t idx2	= int4tx*4;
	int x;
    uint32_t y      = blockIdx.y * blockDim.y + threadIdx.y;
    key += y * 11 * AES_KEYLEN / sizeof(*key);

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
	stageBlock1[tx].uival	    = aesData[AES_BSIZE * bx + tx ];
	tBox0Block[tx].uival		= TBoxi0[tx];
	tBox1Block[tx].uival		= TBoxi1[tx];
	tBox2Block[tx].uival		= TBoxi2[tx];
	tBox3Block[tx].uival		= TBoxi3[tx];
	invSBoxBlock[tx].ubval[0]	= inv_SBox[tx];

	//----------------------------------- 1st stage -----------------------------------

	x = mod4tx;
    stageBlock2[tx].uival = stageBlock1[tx].uival ^ key[x];

	//-------------------------------- end of 1st stage --------------------------------

    uint32_t op[4];

    #pragma unroll
    for (int i = 1; i < 9; i+=2) {
        op[0] = stageBlock2[stageBlockIdx[0]].ubval[0];
        op[1] = stageBlock2[stageBlockIdx[1]].ubval[1];
        op[2] = stageBlock2[stageBlockIdx[2]].ubval[2];
        op[3] = stageBlock2[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        stageBlock1[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

        op[0] = stageBlock1[stageBlockIdx[0]].ubval[0];
        op[1] = stageBlock1[stageBlockIdx[1]].ubval[1];
        op[2] = stageBlock1[stageBlockIdx[2]].ubval[2];
        op[3] = stageBlock1[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        stageBlock2[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];
    }

    op[0] = stageBlock2[stageBlockIdx[0]].ubval[0];
    op[1] = stageBlock2[stageBlockIdx[1]].ubval[1];
    op[2] = stageBlock2[stageBlockIdx[2]].ubval[2];
    op[3] = stageBlock2[stageBlockIdx[3]].ubval[3];

    op[0] = tBox0Block[op[0]].uival;
    op[1] = tBox1Block[op[1]].uival;
    op[2] = tBox2Block[op[2]].uival;
    op[3] = tBox3Block[op[3]].uival;

    x += 4;
    stageBlock1[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

	//----------------------------------- 11th stage -----------------------------------

    op[0] = stageBlock1[stageBlockIdx[0]].ubval[0];
    op[1] = stageBlock1[stageBlockIdx[1]].ubval[1];
    op[2] = stageBlock1[stageBlockIdx[2]].ubval[2];
    op[3] = stageBlock1[stageBlockIdx[3]].ubval[3];

	x += 4;
	stageBlock2[tx].ubval[3] = invSBoxBlock[op[3]].ubval[0]^( key[x]>>24);
	stageBlock2[tx].ubval[2] = invSBoxBlock[op[2]].ubval[0]^( key[x]>>16 & 0x000000FF);
	stageBlock2[tx].ubval[1] = invSBoxBlock[op[1]].ubval[0]^( key[x]>>8  & 0x000000FF);
	stageBlock2[tx].ubval[0] = invSBoxBlock[op[0]].ubval[0]^( key[x]     & 0x000000FF);

	//-------------------------------- end of 11th stage --------------------------------

	aesData[AES_BSIZE * bx + tx] = stageBlock2[tx].uival;
}

__global__
void aesExpand128(uint32_t *keyLeft, uint32_t *keyRight, blk *interleaved,
    blk *separated, uint64_t inWidth) {

    uint32_t bx        = blockIdx.x;
    uint32_t tx        = threadIdx.x;
    uint32_t mod4tx = tx % 4;
    uint32_t int4tx = tx / 4;
    uint32_t idx2    = int4tx * 4;
    int x;
    int expandDir = blockIdx.y;
    uint32_t *key = expandDir == 0 ? keyLeft : keyRight;

    // 18 ms

    uint32_t stageBlockIdx[4] = {
        posIdx_E[mod4tx*4] + idx2,
        posIdx_E[mod4tx*4+1] + idx2,
        posIdx_E[mod4tx*4+2] + idx2,
        posIdx_E[mod4tx*4+3] + idx2
    };

    // 27 ms

    uint64_t parentId =  (bx * AES_BSIZE + tx) / 4;
    uint64_t childId = 2 * parentId + expandDir;

    __shared__ UByte4 stageBlock1[AES_BSIZE];
    __shared__ UByte4 stageBlock2[AES_BSIZE];

    __shared__ UByte4 tBox0Block[256];
    __shared__ UByte4 tBox1Block[256];
    __shared__ UByte4 tBox2Block[256];
    __shared__ UByte4 tBox3Block[256];

    // input caricati in memoria
    stageBlock1[tx].uival = interleaved[(bx*blockDim.x+tx) / 4].data[mod4tx];

    tBox0Block[tx].uival    = TBox0[tx];
    tBox1Block[tx].uival    = TBox1[tx];
    tBox2Block[tx].uival    = TBox2[tx];
    tBox3Block[tx].uival    = TBox3[tx];

    // 42 ms

    if (parentId >= inWidth) return;

    //----------------------------------- 1st stage -----------------------------------

    x = mod4tx;
    stageBlock2[tx].uival = stageBlock1[tx].uival ^ key[x];

    //-------------------------------- end of 1st stage --------------------------------

    uint32_t op[4];

    //50 ms

    #pragma unroll
    for (int i = 1; i < 9; i+=2) {
        op[0] = stageBlock2[stageBlockIdx[0]].ubval[0];
        op[1] = stageBlock2[stageBlockIdx[1]].ubval[1];
        op[2] = stageBlock2[stageBlockIdx[2]].ubval[2];
        op[3] = stageBlock2[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        stageBlock1[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

        op[0] = stageBlock1[stageBlockIdx[0]].ubval[0];
        op[1] = stageBlock1[stageBlockIdx[1]].ubval[1];
        op[2] = stageBlock1[stageBlockIdx[2]].ubval[2];
        op[3] = stageBlock1[stageBlockIdx[3]].ubval[3];

        op[0] = tBox0Block[op[0]].uival;
        op[1] = tBox1Block[op[1]].uival;
        op[2] = tBox2Block[op[2]].uival;
        op[3] = tBox3Block[op[3]].uival;

        x += 4;
        stageBlock2[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];
    }

    op[0] = stageBlock2[stageBlockIdx[0]].ubval[0];
    op[1] = stageBlock2[stageBlockIdx[1]].ubval[1];
    op[2] = stageBlock2[stageBlockIdx[2]].ubval[2];
    op[3] = stageBlock2[stageBlockIdx[3]].ubval[3];

    op[0] = tBox0Block[op[0]].uival;
    op[1] = tBox1Block[op[1]].uival;
    op[2] = tBox2Block[op[2]].uival;
    op[3] = tBox3Block[op[3]].uival;

    x += 4;
    stageBlock1[tx].uival = op[0]^op[1]^op[2]^op[3]^key[x];

    // 229 ms

    //----------------------------------- 11th stage -----------------------------------

    op[0] = stageBlock1[stageBlockIdx[0]].ubval[0];
    op[1] = stageBlock1[stageBlockIdx[1]].ubval[1];
    op[2] = stageBlock1[stageBlockIdx[2]].ubval[2];
    op[3] = stageBlock1[stageBlockIdx[3]].ubval[3];

    x += 4;
    stageBlock2[tx].ubval[3] = tBox1Block[op[3]].ubval[3]^( key[x]>>24);
    stageBlock2[tx].ubval[2] = tBox1Block[op[2]].ubval[3]^( (key[x]>>16) & 0x000000FF);
    stageBlock2[tx].ubval[1] = tBox1Block[op[1]].ubval[3]^( (key[x]>>8)  & 0x000000FF);
    stageBlock2[tx].ubval[0] = tBox1Block[op[0]].ubval[3]^( key[x]       & 0x000000FF);

    //-------------------------------- end of 11th stage --------------------------------

    interleaved[childId].data[mod4tx] = stageBlock2[tx].uival;
    uint64_t offs = expandDir * inWidth;
    separated[parentId + offs].data[mod4tx] = stageBlock2[tx].uival;
} // 290 ms
