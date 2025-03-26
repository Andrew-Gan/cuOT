#ifndef __UTIL_H__
#define __UTIL_H__

// optimization options
#define USE_IMPROVED_AES
#define USE_COALESCED_TREE_EXPANSION
#define USE_COALESCED_NODE_SUMMATION
#define USE_IMPROVED_BIT_TRANSPOSE

#include <cstdint>

enum Role { Sender, Recver };

union blk {
  uint8_t data_8[16];
  uint16_t data_16[8];
  uint32_t data_32[4];
  uint64_t data_64[2];
};

#define BLOCK_BITS (8 * sizeof(blk))

#endif
