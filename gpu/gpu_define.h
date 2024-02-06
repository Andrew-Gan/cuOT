#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdint>

#define AES_BSIZE 256
#define AES_KEYLEN 16
#define AES_PADDING (AES_BSIZE / 4 * 16)

enum Role { Sender, Recver };

struct OTblock {
  uint32_t data[4];
};

using blk = OTblock;

#define BLOCK_BITS (8 * sizeof(blk))

#endif
