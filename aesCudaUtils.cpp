
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

#include "aesCudaUtils.hpp"
#include <string.h>

extern unsigned Rcon[];
extern unsigned SBox[];
extern unsigned LogTable[];
extern unsigned ExpoTable[];

// extern unsigned MODE;

//------------------------------------------help-functions-------------------------------------------------------
unsigned mySbox(unsigned num) {
	return SBox[num];
}

unsigned myXor(unsigned num1, unsigned num2) {
	return num1 ^ num2;
}

//espande la chiave
void expFunc(std::vector<unsigned> &keyArray, std::vector<unsigned> &expKeyArray){
	if ( keyArray.size()!=32 && keyArray.size()!=16 )
		throw std::string("expFunc: key array of wrong dimension");
	if ( expKeyArray.size()!=240 && expKeyArray.size()!=176 )
		throw std::string("expFunc: expanded key array of wrong dimension");

	copy(keyArray.begin(), keyArray.end(), expKeyArray.begin());

	unsigned cycles = (expKeyArray.size()!=240) ? 11 : 8;

	for (unsigned i=1; i<cycles; ++i){
		singleStep(expKeyArray, i);
	}
}

void singleStep(std::vector<unsigned> &expKey, unsigned stepIdx){
	if ( expKey.size()!=240 && expKey.size()!=176 )
		throw std::string("singleStep: expanded key array of wrong dimension");
	if ( stepIdx<1 && stepIdx>11 )
		throw std::string("singleStep: index out of range");

	unsigned num = (expKey.size()!=240) ? 16 : 32;
	unsigned idx = (expKey.size()!=240) ? 16*stepIdx : 32*stepIdx;

	copy(expKey.begin()+(idx)-4, expKey.begin()+(idx),expKey.begin()+(idx));
	rotate(expKey.begin()+(idx), expKey.begin()+(idx)+1, expKey.begin()+(idx)+4);

	transform(expKey.begin()+(idx), expKey.begin()+(idx)+4, expKey.begin()+(idx), mySbox);

	expKey[idx] = expKey[idx] ^ Rcon[stepIdx-1];

	transform(expKey.begin()+(idx), expKey.begin()+(idx)+4, expKey.begin()+(idx)-num, expKey.begin()+(idx), myXor);

	for (unsigned cnt=0; cnt<3; ++cnt){
		copy(expKey.begin()+(idx)+4*cnt, expKey.begin()+(idx)+4*(cnt+1),expKey.begin()+(idx)+(4*(cnt+1)));
		transform(expKey.begin()+(idx)+4*(cnt+1), expKey.begin()+(idx)+4*(cnt+2), expKey.begin()+(idx)-(num-4*(cnt+1)), expKey.begin()+(idx)+4*(cnt+1), myXor);
	}

	if(stepIdx!=7 && expKey.size()!=176){
		copy(expKey.begin()+(idx)+12, expKey.begin()+(idx)+16,expKey.begin()+(idx)+16);
		transform(expKey.begin()+(idx)+16, expKey.begin()+(idx)+20, expKey.begin()+(idx)+16, mySbox);
		transform(expKey.begin()+(idx)+16, expKey.begin()+(idx)+20, expKey.begin()+(idx)-(32-16), expKey.begin()+(idx)+16, myXor);

		for (unsigned cnt=4; cnt<7; ++cnt){
			copy(expKey.begin()+(idx)+4*cnt, expKey.begin()+(idx)+4*(cnt+1),expKey.begin()+(idx)+(4*(cnt+1)));
			transform(expKey.begin()+(idx)+4*(cnt+1), expKey.begin()+(idx)+4*(cnt+2), expKey.begin()+(idx)-(32-4*(cnt+1)), expKey.begin()+(idx)+4*(cnt+1), myXor);
		}
	}
}

//espande la chiave inversa per la decriptazione
void invExpFunc(std::vector<unsigned> &expKey, std::vector<unsigned> &invExpKey){
	if ( expKey.size()!=240 && expKey.size()!=176 )
		throw std::string("invExpFunc: expanded key array of wrong dimension");
	if ( invExpKey.size()!=240 && invExpKey.size()!=176 )
		throw std::string("invExpFunc: inverse expanded key array of wrong dimension");

	std::vector<unsigned> temp(16);

	copy(expKey.begin(), expKey.begin()+16,invExpKey.end()-16);
	copy(expKey.end()-16, expKey.end(),invExpKey.begin());

	unsigned cycles = (expKey.size()!=240) ? 10 : 14;

	for (unsigned cnt=1; cnt<cycles; ++cnt){
		copy(expKey.end()-(16*cnt+16), expKey.end()-(16*cnt), temp.begin());
		invMixColumn(temp);
		copy(temp.begin(), temp.end(), invExpKey.begin()+(16*cnt));
	}
}

void invMixColumn(std::vector<unsigned> &temp){
	if ( temp.size()!=16 )
		throw std::string("invMixColumn: array of wrong dimension");

	std::vector<unsigned> result(4);

	for(unsigned cnt=0; cnt<4; ++cnt){
		result[0] = galoisProd(0x0e, temp[cnt*4]) ^ galoisProd(0x0b, temp[cnt*4+1]) ^ galoisProd(0x0d, temp[cnt*4+2]) ^ galoisProd(0x09, temp[cnt*4+3]);
		result[1] = galoisProd(0x09, temp[cnt*4]) ^ galoisProd(0x0e, temp[cnt*4+1]) ^ galoisProd(0x0b, temp[cnt*4+2]) ^ galoisProd(0x0d, temp[cnt*4+3]);
		result[2] = galoisProd(0x0d, temp[cnt*4]) ^ galoisProd(0x09, temp[cnt*4+1]) ^ galoisProd(0x0e, temp[cnt*4+2]) ^ galoisProd(0x0b, temp[cnt*4+3]);
		result[3] = galoisProd(0x0b, temp[cnt*4]) ^ galoisProd(0x0d, temp[cnt*4+1]) ^ galoisProd(0x09, temp[cnt*4+2]) ^ galoisProd(0x0e, temp[cnt*4+3]);

		copy(result.begin(), result.end(), temp.begin()+(4*cnt));
	}
}

//prodotto di Galois di due numeri
unsigned galoisProd(unsigned a, unsigned b){

	if(a==0 || b==0) return 0;
	else {
		a = LogTable[a];
		b = LogTable[b];
		a = a+b;
		a = a % 255;
		a = ExpoTable[a];
		return a;
	}
}
