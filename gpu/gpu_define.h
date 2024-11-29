#ifndef __UTIL_H__
#define __UTIL_H__

// optimization options
#define USE_IMPROVED_AES
#define USE_COALESCED_TREE_EXPANSION
#define USE_COALESCED_NODE_SUMMATION
#define USE_IMPROVED_BIT_TRANSPOSE

#include <cstdint>

enum Role { Sender, Recver };

struct blk {
  uint32_t data[4];
};

#define BLOCK_BITS (8 * sizeof(blk))

#endif
