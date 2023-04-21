#ifndef __LDPC_H__
#define __LDPC_H__

#include "mytypes.h"

Matrix gen_uniform_random(int numLeaves, int numTrees);
Matrix gen_ldpc(int numLeaves, int numTrees);

#endif
