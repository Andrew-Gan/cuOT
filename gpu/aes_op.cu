#include <utility>
#include "aes_op.h"
#include "sbox_E.h"
#include "sbox_D.h"

union UByte4 {
  uint32_t uival;
  uint8_t ubval[4];
};

__constant__
uint32_t T0c[T_TABLE_SIZE] = {
    0xc66363a5, 0xf87c7c84, 0xee777799, 0xf67b7b8d,
    0xfff2f20d, 0xd66b6bbd, 0xde6f6fb1, 0x91c5c554,
    0x60303050, 0x02010103, 0xce6767a9, 0x562b2b7d,
    0xe7fefe19, 0xb5d7d762, 0x4dababe6, 0xec76769a,
    0x8fcaca45, 0x1f82829d, 0x89c9c940, 0xfa7d7d87,
    0xeffafa15, 0xb25959eb, 0x8e4747c9, 0xfbf0f00b,
    0x41adadec, 0xb3d4d467, 0x5fa2a2fd, 0x45afafea,
    0x239c9cbf, 0x53a4a4f7, 0xe4727296, 0x9bc0c05b,
    0x75b7b7c2, 0xe1fdfd1c, 0x3d9393ae, 0x4c26266a,
    0x6c36365a, 0x7e3f3f41, 0xf5f7f702, 0x83cccc4f,
    0x6834345c, 0x51a5a5f4, 0xd1e5e534, 0xf9f1f108,
    0xe2717193, 0xabd8d873, 0x62313153, 0x2a15153f,
    0x0804040c, 0x95c7c752, 0x46232365, 0x9dc3c35e,
    0x30181828, 0x379696a1, 0x0a05050f, 0x2f9a9ab5,
    0x0e070709, 0x24121236, 0x1b80809b, 0xdfe2e23d,
    0xcdebeb26, 0x4e272769, 0x7fb2b2cd, 0xea75759f,
    0x1209091b, 0x1d83839e, 0x582c2c74, 0x341a1a2e,
    0x361b1b2d, 0xdc6e6eb2, 0xb45a5aee, 0x5ba0a0fb,
    0xa45252f6, 0x763b3b4d, 0xb7d6d661, 0x7db3b3ce,
    0x5229297b, 0xdde3e33e, 0x5e2f2f71, 0x13848497,
    0xa65353f5, 0xb9d1d168, 0x00000000, 0xc1eded2c,
    0x40202060, 0xe3fcfc1f, 0x79b1b1c8, 0xb65b5bed,
    0xd46a6abe, 0x8dcbcb46, 0x67bebed9, 0x7239394b,
    0x944a4ade, 0x984c4cd4, 0xb05858e8, 0x85cfcf4a,
    0xbbd0d06b, 0xc5efef2a, 0x4faaaae5, 0xedfbfb16,
    0x864343c5, 0x9a4d4dd7, 0x66333355, 0x11858594,
    0x8a4545cf, 0xe9f9f910, 0x04020206, 0xfe7f7f81,
    0xa05050f0, 0x783c3c44, 0x259f9fba, 0x4ba8a8e3,
    0xa25151f3, 0x5da3a3fe, 0x804040c0, 0x058f8f8a,
    0x3f9292ad, 0x219d9dbc, 0x70383848, 0xf1f5f504,
    0x63bcbcdf, 0x77b6b6c1, 0xafdada75, 0x42212163,
    0x20101030, 0xe5ffff1a, 0xfdf3f30e, 0xbfd2d26d,
    0x81cdcd4c, 0x180c0c14, 0x26131335, 0xc3ecec2f,
    0xbe5f5fe1, 0x359797a2, 0x884444cc, 0x2e171739,
    0x93c4c457, 0x55a7a7f2, 0xfc7e7e82, 0x7a3d3d47,
    0xc86464ac, 0xba5d5de7, 0x3219192b, 0xe6737395,
    0xc06060a0, 0x19818198, 0x9e4f4fd1, 0xa3dcdc7f,
    0x44222266, 0x542a2a7e, 0x3b9090ab, 0x0b888883,
    0x8c4646ca, 0xc7eeee29, 0x6bb8b8d3, 0x2814143c,
    0xa7dede79, 0xbc5e5ee2, 0x160b0b1d, 0xaddbdb76,
    0xdbe0e03b, 0x64323256, 0x743a3a4e, 0x140a0a1e,
    0x924949db, 0x0c06060a, 0x4824246c, 0xb85c5ce4,
    0x9fc2c25d, 0xbdd3d36e, 0x43acacef, 0xc46262a6,
    0x399191a8, 0x319595a4, 0xd3e4e437, 0xf279798b,
    0xd5e7e732, 0x8bc8c843, 0x6e373759, 0xda6d6db7,
    0x018d8d8c, 0xb1d5d564, 0x9c4e4ed2, 0x49a9a9e0,
    0xd86c6cb4, 0xac5656fa, 0xf3f4f407, 0xcfeaea25,
    0xca6565af, 0xf47a7a8e, 0x47aeaee9, 0x10080818,
    0x6fbabad5, 0xf0787888, 0x4a25256f, 0x5c2e2e72,
    0x381c1c24, 0x57a6a6f1, 0x73b4b4c7, 0x97c6c651,
    0xcbe8e823, 0xa1dddd7c, 0xe874749c, 0x3e1f1f21,
    0x964b4bdd, 0x61bdbddc, 0x0d8b8b86, 0x0f8a8a85,
    0xe0707090, 0x7c3e3e42, 0x71b5b5c4, 0xcc6666aa,
    0x904848d8, 0x06030305, 0xf7f6f601, 0x1c0e0e12,
    0xc26161a3, 0x6a35355f, 0xae5757f9, 0x69b9b9d0,
    0x17868691, 0x99c1c158, 0x3a1d1d27, 0x279e9eb9,
    0xd9e1e138, 0xebf8f813, 0x2b9898b3, 0x22111133,
    0xd26969bb, 0xa9d9d970, 0x078e8e89, 0x339494a7,
    0x2d9b9bb6, 0x3c1e1e22, 0x15878792, 0xc9e9e920,
    0x87cece49, 0xaa5555ff, 0x50282878, 0xa5dfdf7a,
    0x038c8c8f, 0x59a1a1f8, 0x09898980, 0x1a0d0d17,
    0x65bfbfda, 0xd7e6e631, 0x844242c6, 0xd06868b8,
    0x824141c3, 0x299999b0, 0x5a2d2d77, 0x1e0f0f11,
    0x7bb0b0cb, 0xa85454fc, 0x6dbbbbd6, 0x2c16163a
};
// Shifts
#define ROTR_1B 0x00004321
#define ROTR_2B 0x00005432
#define ROTR_3B 0x00006543
#define ROTR(x, sel) __byte_perm(x, x, sel)
#define GET_BYTE_0 0x00004560
#define GET_BYTE_1 0x00004561
#define GET_BYTE_2 0x00004562
#define GET_BYTE_3 0x00004563
#define GET_BYTE(x, sel) __byte_perm(x, 0, sel)
#define FLIP_ENDIANESS(x) __byte_perm(x, x, 0x0123)

__global__
void aesEncrypt128(uint32_t* rk, uint32_t* data) {
    uint32_t s[4];
    uint32_t nexts[4];

    uint32_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warpID = threadIdx.x / GPU_SHARED_MEM_BANK;
    uint32_t warpThreadIndex = threadIdx.x % GPU_SHARED_MEM_BANK;

    // Thread is responsible for this block
    uint32_t* myData = data + threadID * (AES_BLOCK_SIZE / sizeof(*data));

    __shared__ uint32_t t0S[T_TABLE_SIZE][GPU_SHARED_MEM_BANK];
    __shared__ uint32_t rkS[AES_RK_SIZE / sizeof(*rk)];
    uint32_t* currRk = rkS;

    // Copy over T table and round key
    for(uint32_t i = 0; i < GPU_SHARED_MEM_BANK/(blockDim.x/T_TABLE_SIZE); i++) {
        uint32_t tableLoc = warpID + i * (T_TABLE_SIZE / (GPU_SHARED_MEM_BANK/(blockDim.x/T_TABLE_SIZE)));
        t0S[tableLoc][warpThreadIndex] = T0c[tableLoc];
    }
    if(threadIdx.x < AES_RK_SIZE / sizeof(*rk)) {
        rkS[threadIdx.x] = rk[threadIdx.x];
    }
    __syncthreads();

    // Encrypt
    // Initial Step 
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        s[i] = FLIP_ENDIANESS(myData[i] ^ currRk[i]);
    }
    // Middle rounds
    #pragma unroll
    for(int i = 0; i < AES_NUM_ROUNDS-1; i++) {
        currRk += 4;
        nexts[0] = t0S[GET_BYTE(s[0], GET_BYTE_3)][warpThreadIndex] ^
            ROTR(t0S[GET_BYTE(s[1], GET_BYTE_2)][warpThreadIndex], ROTR_1B) ^
            ROTR(t0S[GET_BYTE(s[2], GET_BYTE_1)][warpThreadIndex], ROTR_2B) ^
            ROTR(t0S[GET_BYTE(s[3], GET_BYTE_0)][warpThreadIndex], ROTR_3B);
        nexts[1] = t0S[GET_BYTE(s[1], GET_BYTE_3)][warpThreadIndex] ^
            ROTR(t0S[GET_BYTE(s[2], GET_BYTE_2)][warpThreadIndex], ROTR_1B) ^
            ROTR(t0S[GET_BYTE(s[3], GET_BYTE_1)][warpThreadIndex], ROTR_2B) ^
            ROTR(t0S[GET_BYTE(s[0], GET_BYTE_0)][warpThreadIndex], ROTR_3B);
        nexts[2] = t0S[GET_BYTE(s[2], GET_BYTE_3)][warpThreadIndex] ^
            ROTR(t0S[GET_BYTE(s[3], GET_BYTE_2)][warpThreadIndex], ROTR_1B) ^
            ROTR(t0S[GET_BYTE(s[0], GET_BYTE_1)][warpThreadIndex], ROTR_2B) ^
            ROTR(t0S[GET_BYTE(s[1], GET_BYTE_0)][warpThreadIndex], ROTR_3B);
        nexts[3] = t0S[GET_BYTE(s[3], GET_BYTE_3)][warpThreadIndex] ^
            ROTR(t0S[GET_BYTE(s[0], GET_BYTE_2)][warpThreadIndex], ROTR_1B) ^
            ROTR(t0S[GET_BYTE(s[1], GET_BYTE_1)][warpThreadIndex], ROTR_2B) ^
            ROTR(t0S[GET_BYTE(s[2], GET_BYTE_0)][warpThreadIndex], ROTR_3B);
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            s[j] = nexts[j] ^ FLIP_ENDIANESS(currRk[j]);
        }
    }
    // Last round, get S-box from T-box
    currRk += 4;
    nexts[0] = __byte_perm(t0S[s[0] >> 24][warpThreadIndex], 0, 0x1456) ^
        __byte_perm(t0S[GET_BYTE(s[1], GET_BYTE_2)][warpThreadIndex], 0, 0x4156) ^
        (t0S[GET_BYTE(s[2], GET_BYTE_1)][warpThreadIndex] & 0xff00) ^
        __byte_perm(t0S[GET_BYTE(s[3], GET_BYTE_0)][warpThreadIndex], 0, 0x4561) ^
        FLIP_ENDIANESS(currRk[0]);
    nexts[1] = __byte_perm(t0S[s[1] >> 24][warpThreadIndex], 0, 0x1456) ^
        __byte_perm(t0S[GET_BYTE(s[2], GET_BYTE_2)][warpThreadIndex], 0, 0x4156) ^
        (t0S[GET_BYTE(s[3], GET_BYTE_1)][warpThreadIndex] & 0xff00) ^
        __byte_perm(t0S[GET_BYTE(s[0], GET_BYTE_0)][warpThreadIndex], 0, 0x4561) ^
        FLIP_ENDIANESS(currRk[1]);
    nexts[2] = __byte_perm(t0S[s[2] >> 24][warpThreadIndex], 0, 0x1456) ^
        __byte_perm(t0S[GET_BYTE(s[3], GET_BYTE_2)][warpThreadIndex], 0, 0x4156) ^
        (t0S[GET_BYTE(s[0], GET_BYTE_1)][warpThreadIndex] & 0xff00) ^
        __byte_perm(t0S[GET_BYTE(s[1], GET_BYTE_0)][warpThreadIndex], 0, 0x4561) ^
        FLIP_ENDIANESS(currRk[2]);
    nexts[3] = __byte_perm(t0S[s[3] >> 24][warpThreadIndex], 0, 0x1456) ^
        __byte_perm(t0S[GET_BYTE(s[0], GET_BYTE_2)][warpThreadIndex], 0, 0x4156) ^
        (t0S[GET_BYTE(s[1], GET_BYTE_1)][warpThreadIndex] & 0xff00) ^
        __byte_perm(t0S[GET_BYTE(s[2], GET_BYTE_0)][warpThreadIndex], 0, 0x4561) ^
        FLIP_ENDIANESS(currRk[3]);

    // Write back
    #pragma unroll
    for(int i = 0; i < 4; i++)
        myData[i] = FLIP_ENDIANESS(nexts[i]);
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
void aesExpand128(uint32_t *keyLeft, uint32_t *keyRight, blk *interleaved_in,
    blk *interleaved_out, blk *separated, uint64_t inWidth) {

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
    stageBlock1[tx].uival = interleaved_in[(bx*blockDim.x+tx) / 4].data[mod4tx];

    tBox0Block[tx].uival    = TBox0[tx];
    tBox1Block[tx].uival    = TBox1[tx];
    tBox2Block[tx].uival    = TBox2[tx];
    tBox3Block[tx].uival    = TBox3[tx];

    if (parentId >= inWidth) return;

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

    interleaved_out[childId].data[mod4tx] = stageBlock2[tx].uival;
    uint64_t offs = expandDir * inWidth;
    separated[parentId + offs].data[mod4tx] = stageBlock2[tx].uival;
}
