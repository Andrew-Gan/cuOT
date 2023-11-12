#ifndef __MATMULT_H__
#define __MATMULT_H__

__global__
void make_block(blk *blocks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockDim.y;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    blocks[d*i+j].data[0] = 4*i;
    blocks[d*i+j].data[2] = j;
}

__global__
void lpn_single_row(uint32_t *r, int d, int k, blk *nn, blk *kk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    blk tmp_nn = nn[i];
    for (int j = 0; j < d; ++j) {
        // decrease k such that fits in shared memory
        // increase t due to massive parallelisation
        // keep n constant based on required OTs
        blk tmp_kk = kk[r[d*i+j] % k];

        tmp_nn.data[0] ^= tmp_kk.data[0];
        tmp_nn.data[1] ^= tmp_kk.data[1];
        tmp_nn.data[2] ^= tmp_kk.data[2];
        tmp_nn.data[3] ^= tmp_kk.data[3];
    }
    nn[i] = tmp_nn;
}

#endif
