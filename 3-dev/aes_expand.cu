
/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


/**
	@author Svetlin Manavski <svetlin@manavski.com>
 */

/* aes encryption operation:
 * Device code.
 *
 */

#include "util.h"
#include "sbox_E.h"
#include "aes_expand.h"

__global__
void aesExpand128(uint32_t *keyLeft, uint32_t *keyRight, OTblock *interleaved,
	OTblock *left, OTblock *right, OTblock *inData, uint64_t width) {
		
	uint32_t bx		= blockIdx.x;
    uint32_t tx		= threadIdx.x;
    uint32_t mod4tx = tx % 4;
    uint32_t int4tx = tx / 4;
    uint32_t idx2	= int4tx * 4;
	int x;
	int expandDir = blockIdx.y;
	uint32_t *aesKey = expandDir == 0 ? keyLeft : keyRight;

	int elemPerNode = sizeof(OTblock) / 4;
	uint64_t parentId =  (bx * AES_BSIZE + tx) / elemPerNode;
	uint64_t childId = 2 * parentId + expandDir;

    __shared__ UByte4 stageBlock1[AES_BSIZE];
	__shared__ UByte4 stageBlock2[AES_BSIZE];

	__shared__ UByte4 tBox0Block[256];
	__shared__ UByte4 tBox1Block[256];
	__shared__ UByte4 tBox2Block[256];
	__shared__ UByte4 tBox3Block[256];

	// input caricati in memoria
	stageBlock1[tx].uival	= inData[(blockDim.x * bx + tx) / 4].data[mod4tx];

	if (tx < 256) {
		tBox0Block[tx].uival	= TBox0[tx];
		tBox1Block[tx].uival	= TBox1[tx];
		tBox2Block[tx].uival	= TBox2[tx];
		tBox3Block[tx].uival	= TBox3[tx];
	}

	__syncthreads();

	//----------------------------------- 1st stage -----------------------------------

	x = mod4tx;
    stageBlock2[tx].uival = stageBlock1[tx].uival ^ aesKey[x];

	__syncthreads();

	//-------------------------------- end of 1st stage --------------------------------


	//----------------------------------- 2nd stage -----------------------------------

    uint32_t op1 = stageBlock2[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	uint32_t op2 = stageBlock2[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	uint32_t op3 = stageBlock2[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	uint32_t op4 = stageBlock2[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+4;
	 stageBlock1[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 2nd stage --------------------------------

	//----------------------------------- 3th stage -----------------------------------

    op1 = stageBlock1[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+8;
	 stageBlock2[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 3th stage --------------------------------

	//----------------------------------- 4th stage -----------------------------------

    op1 = stageBlock2[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock2[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock2[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock2[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+12;
	 stageBlock1[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 4th stage --------------------------------

	//----------------------------------- 5th stage -----------------------------------

    op1 = stageBlock1[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+16;
	 stageBlock2[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 5th stage --------------------------------

	//----------------------------------- 6th stage -----------------------------------

    op1 = stageBlock2[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock2[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock2[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock2[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+20;
	 stageBlock1[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 6th stage --------------------------------

	//----------------------------------- 7th stage -----------------------------------

    op1 = stageBlock1[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+24;
	stageBlock2[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 7th stage --------------------------------

	//----------------------------------- 8th stage -----------------------------------

    op1 = stageBlock2[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock2[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock2[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock2[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+28;
	stageBlock1[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 8th stage --------------------------------

	//----------------------------------- 9th stage -----------------------------------

    op1 = stageBlock1[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+32;
	stageBlock2[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 9th stage --------------------------------

	//----------------------------------- 10th stage -----------------------------------

    op1 = stageBlock2[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock2[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock2[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock2[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+36;
	stageBlock1[tx].uival = op1^op2^op3^op4^aesKey[x];

	__syncthreads();

	//-------------------------------- end of 10th stage --------------------------------

	//----------------------------------- 11th stage -----------------------------------

    op1 = stageBlock1[posIdx_E[mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_E[mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_E[mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_E[mod4tx*4+3] + idx2].ubval[3];

	x = mod4tx+40;


	stageBlock2[tx].ubval[3] = tBox1Block[op4].ubval[3]^( aesKey[x]>>24);
	stageBlock2[tx].ubval[2] = tBox1Block[op3].ubval[3]^( (aesKey[x]>>16) & 0x000000FF);
	stageBlock2[tx].ubval[1] = tBox1Block[op2].ubval[3]^( (aesKey[x]>>8)  & 0x000000FF);
	stageBlock2[tx].ubval[0] = tBox1Block[op1].ubval[3]^( aesKey[x]       & 0x000000FF);

	__syncthreads();

	//-------------------------------- end of 11th stage --------------------------------

	if (childId < width) {
		interleaved[childId].data[tx % elemPerNode] = stageBlock2[tx].uival;
	}
	OTblock *separated = expandDir == 0 ? left : right;
	separated[parentId].data[mod4tx] = stageBlock2[tx].uival;
}
