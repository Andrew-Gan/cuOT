#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"

std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;

SilentOTSender::SilentOTSender(SilentOTConfig config) : SilentOT(config) {
  blk seed_h, delta_h;
  for (int i = 0; i < 4; i++) delta_h.data[i] = rand();

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    std::vector<cudaEvent_t> &events = expandEvents[gpu];
    events.resize(depth);
    for (uint64_t i = 0; i < depth; i++) {
      cudaEventCreate(&events.at(i));
    }
    silentOTSenders[config.id] = this;
    while(silentOTRecvers[config.id] == nullptr);
    other = silentOTRecvers[config.id];

    if (gpu == 0) {
      fullVector[gpu] = new Mat({numOT, 1});
      fullVector[gpu]->resize({numOT / NGPU, 1});
    }
    else
      fullVector[gpu] = new Mat({numOT / NGPU, 1});
    buffer[gpu] = new Mat({numOT / NGPU, 1});
    cudaMalloc(&delta[gpu], sizeof(**delta));
    cudaMemcpy(delta[gpu], &delta_h, sizeof(**delta), cudaMemcpyHostToDevice);
    for (uint64_t t = 0; t < mConfig.nTree / NGPU; t++) {
      for (int i = 0; i < 4; i++) seed_h.data[i] = rand();
      fullVector[gpu]->set(seed_h, {t, 0});
    }
    separated[gpu].resize({numOT / NGPU});
    switch (mConfig.expander) {
      case AesExpand_t:
        expander[gpu] = new AesExpand(mConfig.leftKey, mConfig.rightKey);
    }

    uint64_t rowsPerGPU = (BLOCK_BITS + NGPU - 1) / NGPU;
    b64[gpu].resize({rowsPerGPU, 2 * numOT / BLOCK_BITS});
    b64[gpu].clear();
    switch (mConfig.compressor) {
      case QuasiCyclic_t:
        lpn[gpu] = new QuasiCyclic(Sender, 2 * numOT, numOT, BLOCK_BITS / NGPU);
    }
  }
}

SilentOTSender::~SilentOTSender() {
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    delete expander[gpu];
    delete fullVector[gpu];
    delete buffer[gpu];
    delete lpn[gpu];
    for (uint64_t d = 0; d < depth; d++)
      cudaEventDestroy(expandEvents[gpu].at(d));
    cudaFree(delta[gpu]);
  }
}

void SilentOTSender::base_ot() {
  Log::mem(Sender, BaseOT);
  std::vector<std::future<std::array<Mat, 2>>> workers[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    for (int d = 0; d < depth; d++) {
      workers[gpu].push_back(std::async([d, gpu, this]() {
        cudaSetDevice(gpu);
        switch (mConfig.baseOT) {
          case SimplestOT_t:
            return SimplestOT(Sender, gpu*this->depth+d, mConfig.nTree / NGPU).send();
        }
        return std::array<Mat, 2>();
      }));
    }
  }

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    for (auto &worker : workers[gpu]) {
      std::array<Mat, 2> res = worker.get();
      m0[gpu].push_back(res[0]);
      m1[gpu].push_back(res[1]);
    }
    m0[gpu].push_back(m0[gpu].back());
    m1[gpu].push_back(m1[gpu].back());
  }
  Log::mem(Sender, BaseOT);
}

void SilentOTSender::pprf_expand() {
  Log::mem(Sender, SeedExp);
  int treePerGPU = (mConfig.nTree + NGPU - 1) / NGPU;
  Mat *input[NGPU];
  Mat *output[NGPU];

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    input[gpu] = buffer[gpu];
    output[gpu] = fullVector[gpu];
    for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
      std::swap(input[gpu], output[gpu]);
      expander[gpu]->expand(*(input[gpu]), *(output[gpu]), separated[gpu], treePerGPU*inWidth);
      separated[gpu].sum(2 * treePerGPU, inWidth);
      m0[gpu].at(d).xor_d(separated[gpu], 0);
      m1[gpu].at(d).xor_d(separated[gpu], treePerGPU);

      if (d == depth-1) {
        m0[gpu].at(d+1).xor_d(separated[gpu], treePerGPU);
        m1[gpu].at(d+1).xor_d(separated[gpu], 0);
        m0[gpu].at(d+1).xor_scalar(delta[gpu]);
        m1[gpu].at(d+1).xor_scalar(delta[gpu]);
      }
      cudaEventRecord(expandEvents[gpu].at(d));
    }
  }
  other->expandReady = true;
  Log::mem(Sender, SeedExp);

  for (int gpu = 0; gpu < NGPU; gpu++) {
    fullVector[gpu] = output[gpu];
    buffer[gpu] = input[gpu];
    cudaSetDevice(gpu);
    cudaDeviceSynchronize();
  }
}

void SilentOTSender::lpn_compress() {
  Log::mem(Sender, LPN);

  uint64_t rowsPerGPU = (BLOCK_BITS + NGPU - 1) / NGPU;
  cudaStream_t *s = new cudaStream_t[rowsPerGPU];
  for (uint64_t r = 0; r < rowsPerGPU; r++) {
    cudaStreamCreate(&s[r]);
  }

  for(int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    fullVector[gpu]->bit_transpose();
  }
  for (int des = 0; des < NGPU; des++) {
    for (int src = 0; src < NGPU; src++) {
      cudaMemcpy2DPeerAsync(
        b64[des].data({0, src*fullVector[src]->dim(1)}), b64[des].dim(1)*sizeof(blk), des,
        fullVector[src]->data({des * rowsPerGPU, 0}), fullVector[src]->dim(1)*sizeof(blk), src,
        fullVector[src]->dim(1)*sizeof(blk), rowsPerGPU, s
      );
    }
  }
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    lpn[gpu]->encode_dense(b64[gpu]);
  }
  cudaSetDevice(0);
  fullVector[0]->resize({BLOCK_BITS, numOT / BLOCK_BITS});

  uint64_t dWidth = fullVector[0]->dim(1)*sizeof(blk);
  uint64_t sWidth = b64[0].dim(1)*sizeof(blk);
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaMemcpy2DPeerAsync(
      fullVector[0]->data({gpu * rowsPerGPU, 0}), dWidth, 0,
      b64[gpu].data(), sWidth, gpu, dWidth, rowsPerGPU, s
    );
  }
  fullVector[0]->bit_transpose();

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    cudaDeviceSynchronize();
  }
  for (uint64_t r = 0; r < rowsPerGPU; r++) {
    cudaStreamDestroy(s[r]);
  }
}
