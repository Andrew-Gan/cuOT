#ifndef __LDPC_H__
#define __LDPC_H__

#include "mytypes.h"

void print_matrix(Matrix& mat);

Matrix generate_ldpc(int numLeaves, int numTrees);

#endif
