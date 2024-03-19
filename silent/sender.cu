#include "roles.h"
#include <future>

#include "logger.h"
#include "gpu_ops.h"

std::array<std::atomic<SilentOTSender*>, 16> silentOTSenders;

SilentOTSender::SilentOTSender(SilentOTConfig config) : SilentOT(config) {
  blk seed_h, delta_h;
  for (int i = 0; i < 4; i++) delta_h.data[i] = rand();
  silentOTSenders[mConfig.id] = this;
  while(silentOTRecvers[mConfig.id] == nullptr);
  other = silentOTRecvers[mConfig.id];

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    std::vector<cudaEvent_t> &events = expandEvents[gpu];
    events.resize(depth);

    for (uint64_t i = 0; i < depth; i++)
      cudaEventCreate(&events.at(i));

    fullVector[gpu] = new Mat({2 * numOT, 1});
    buffer[gpu] = new Mat({numOT, 1});
    cudaMalloc(&delta[gpu], sizeof(**delta));
    cudaMemcpy(delta[gpu], &delta_h, sizeof(**delta), cudaMemcpyHostToDevice);

    for (uint64_t t = 0; t < mConfig.nTree; t++) {
      for (int i = 0; i < 4; i++) seed_h.data[i] = rand();
      fullVector[gpu]->set(seed_h, {t, 0});
    }
    separated[gpu].resize({numOT});
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
    delete fullVector[gpu];
    delete buffer[gpu];
    delete expander[gpu];
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
            return SimplestOT(Sender, gpu*this->depth+d, mConfig.nTree).send();
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
  Mat *input[NGPU];
  Mat *output[NGPU];

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    input[gpu] = buffer[gpu];
    output[gpu] = fullVector[gpu];
    for (uint64_t d = 0, inWidth = 1; d < depth; d++, inWidth *= 2) {
      std::swap(input[gpu], output[gpu]);
      expander[gpu]->expand(*(input[gpu]), *(output[gpu]), separated[gpu], mConfig.nTree*inWidth);
      separated[gpu].sum(2 * mConfig.nTree, inWidth);
      m0[gpu].at(d).xor_d(separated[gpu], 0);
      m1[gpu].at(d).xor_d(separated[gpu], mConfig.nTree);

      if (d == depth-1) {
        m0[gpu].at(d+1).xor_d(separated[gpu], mConfig.nTree);
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
  for(int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    fullVector[gpu]->bit_transpose();
  }
  Span *span[NGPU];
  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    span[gpu] = new Span(fullVector[gpu], {gpu*rowsPerGPU, 0}, {(gpu+1)*rowsPerGPU, 0});
    lpn[gpu]->encode_dense(span[gpu]);
    fullVector[gpu].resize({fullVector[gpu].dim(0), mOut / BLOCK_BITS});
  }
  cudaSetDevice(0);
  fullVector[0]->resize({BLOCK_BITS, numOT / BLOCK_BITS});
  for (int gpu = 1; gpu < NGPU; gpu++) {
    cudaMemcpyPeerAsync(
      fullVector[0]->data({gpu * rowsPerGPU, 0}), 0,
      fullVector[gpu]->data(), gpu, fullVector[gpu]->size_bytes()
    );
  }
  fullVector[0]->bit_transpose();

  for (int gpu = 0; gpu < NGPU; gpu++) {
    cudaSetDevice(gpu);
    cudaDeviceSynchronize();
  }
}
