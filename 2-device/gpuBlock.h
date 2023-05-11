#ifndef __GPU_BLOCK_H__
#define __GPU_BLOCK_H__

class GPUBlock {
    GPUBlock();
    GPUBlock(size_t n);
    GPUBlock(const GPUBlock &blk);
    virtual ~GPUBlock();
    uint8_t *m_data_d = nullptr;
    size_t m_nBytes = 0;
    GPUBlock operator^(const GPUBlock &rhs);
    GPUBlock& operator=(const GPUBlock &rhs);
    bool operator==(const GPUBlock &rhs);
    uint8_t* operator[](int index);
    void set(uint32_t rhs);
};

#endif
