#include "rsa.h"
#include "basic_op.h"


Rsa::Rsa() {
}

Rsa::Rsa(uint1024_t key_e, uint1024_t key_n) {
  e = key_e;
  n = key_n;
}

void Rsa::encrypt(GPUBlock m) {
  EventLog::start(RsaEncrypt);
  uint32_t *casted = (uint32_t*) m.data_d;
  for (size_t i = 0; i < m.nBytes / 4; i++) {
    printf("loop %lu out of %lu\n", i, (m.nBytes-1) / 4 + 1);
    bool rsa1024(uint64_t res[], uint64_t data[], uint64_t expo[],uint64_t key[])
    cudaDeviceSynchronize();
  }
  EventLog::end(RsaEncrypt);
}

void Rsa::decrypt(GPUBlock m) {
  EventLog::start(RsaDecrypt);
  uint32_t *casted = (uint32_t*) m.data_d;
  for (size_t i = 0; i < m.nBytes / 4; i++) {
    printf("loop %lu out of %lu\n", i, (m.nBytes-1) / 4 + 1);
    chinese_rem_theorem_gpu<<<1, 1>>>(&casted[i], d, p, q, d_p, d_q, q_inv);
    cudaDeviceSynchronize();
  }
  EventLog::end(RsaDecrypt);
}
