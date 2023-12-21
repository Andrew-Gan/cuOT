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

void check_alloc(blk *ptr);
void check_call(const char *msg);
bool check_rot(Vec &m0, Vec &m1, Vec &mc, uint8_t *c);
bool check_cot(Vec &full, Vec &punc, uint64_t *choice, blk delta);

#endif
