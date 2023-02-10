
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

/* aes decryption operation:
 * Device code.
 *
 */

#ifndef _AESDECRYPT128_KERNEL_H_
#define _AESDECRYPT128_KERNEL_H_

#include <stdio.h>

// Thread block size
#define BSIZE 256

__global__ void aesDecrypt128(cudaTextureObject_t texDKey128, unsigned * result, unsigned * inData, int inputSize)
{
	unsigned bx		= blockIdx.x;
    unsigned tx		= threadIdx.x;
    unsigned mod4tx = tx%4;
    unsigned int4tx = tx/4;
    unsigned idx2	= int4tx*4;
	int x;
	unsigned keyElem;

    __shared__ UByte4 stageBlock1[BSIZE];
	__shared__ UByte4 stageBlock2[BSIZE];
    __shared__ UByte4 tBox0Block[BSIZE];
    __shared__ UByte4 tBox1Block[BSIZE];
    __shared__ UByte4 tBox2Block[BSIZE];
    __shared__ UByte4 tBox3Block[BSIZE];
    __shared__ UByte4 invSBoxBlock[BSIZE];

	// input caricati in memoria
	stageBlock1[tx].uival	= inData[BSIZE * bx + tx ];
	tBox0Block[tx].uival		= TBoxi0[tx];
	tBox1Block[tx].uival		= TBoxi1[tx];
	tBox2Block[tx].uival		= TBoxi2[tx];
	tBox3Block[tx].uival		= TBoxi3[tx];
	invSBoxBlock[tx].ubval[0]	= inv_SBox[tx];

	__syncthreads();


	//----------------------------------- 1st stage -----------------------------------

	x = mod4tx;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
    stageBlock2[tx].uival = stageBlock1[tx].uival ^ keyElem;

	__syncthreads();

	//-------------------------------- end of 1st stage --------------------------------


	//----------------------------------- 2nd stage -----------------------------------

    unsigned op1 = stageBlock2[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	unsigned op2 = stageBlock2[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	unsigned op3 = stageBlock2[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	unsigned op4 = stageBlock2[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+4;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	stageBlock1[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 2nd stage --------------------------------

	//----------------------------------- 3th stage -----------------------------------

    op1 = stageBlock1[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+8;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	 stageBlock2[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 3th stage --------------------------------

	//----------------------------------- 4th stage -----------------------------------

    op1 = stageBlock2[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock2[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock2[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock2[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+12;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	 stageBlock1[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 4th stage --------------------------------

	//----------------------------------- 5th stage -----------------------------------

    op1 = stageBlock1[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+16;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	 stageBlock2[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 5th stage --------------------------------

	//----------------------------------- 6th stage -----------------------------------

    op1 = stageBlock2[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock2[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock2[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock2[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+20;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	 stageBlock1[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 6th stage --------------------------------

	//----------------------------------- 7th stage -----------------------------------

    op1 = stageBlock1[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+24;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	stageBlock2[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 7th stage --------------------------------

	//----------------------------------- 8th stage -----------------------------------

    op1 = stageBlock2[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock2[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock2[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock2[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+28;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	stageBlock1[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 8th stage --------------------------------

	//----------------------------------- 9th stage -----------------------------------

    op1 = stageBlock1[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+32;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	stageBlock2[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 9th stage --------------------------------

	//----------------------------------- 10th stage -----------------------------------

    op1 = stageBlock2[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock2[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock2[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock2[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	op1 = tBox0Block[op1].uival;

    op2 = tBox1Block[op2].uival;

    op3 = tBox2Block[op3].uival;

    op4 = tBox3Block[op4].uival;

	x = mod4tx+36;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);
	stageBlock1[tx].uival = op1^op2^op3^op4^keyElem;

	__syncthreads();

	//-------------------------------- end of 10th stage --------------------------------

	//----------------------------------- 11th stage -----------------------------------

    op1 = stageBlock1[posIdx_D[16 + mod4tx*4]   + idx2].ubval[0];
	op2 = stageBlock1[posIdx_D[16 + mod4tx*4+1] + idx2].ubval[1];
	op3 = stageBlock1[posIdx_D[16 + mod4tx*4+2] + idx2].ubval[2];
	op4 = stageBlock1[posIdx_D[16 + mod4tx*4+3] + idx2].ubval[3];

	x = mod4tx+40;
	keyElem = tex1Dfetch<unsigned>(texDKey128, x);

	stageBlock2[tx].ubval[3] = invSBoxBlock[op4].ubval[0]^( keyElem>>24);
	stageBlock2[tx].ubval[2] = invSBoxBlock[op3].ubval[0]^( keyElem>>16 & 0x000000FF);
	stageBlock2[tx].ubval[1] = invSBoxBlock[op2].ubval[0]^( keyElem>>8  & 0x000000FF);
	stageBlock2[tx].ubval[0] = invSBoxBlock[op1].ubval[0]^( keyElem     & 0x000000FF);

	__syncthreads();

	//-------------------------------- end of 11th stage --------------------------------

	result[BSIZE * bx + tx] = stageBlock2[tx].uival;
	// end of AES

}

#endif // #ifndef _AESDECRYPT_KERNEL_H_

