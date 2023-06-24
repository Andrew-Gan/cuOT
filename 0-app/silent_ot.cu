#include "hash.h"
#include "rand.h"
#include "aes.h"
#include "simplest_ot.h"
#include "silent_ot.h"
#include "basic_op.h"
#include <future>

std::array<std::atomic<SilentOT*>, 100> silentOTSenders;
std::array<std::atomic<SilentOT*>, 100> silentOTRecvers;

SilentOT::SilentOT(Role myrole, int myid, int logOT, int numTrees, uint64_t *mychoices) {
  role = myrole;
  id = myid;
  choices = mychoices;

  if (role == Sender) {
    silentOTSenders[id] = this;
    while(silentOTRecvers[id] == nullptr);
    other = silentOTRecvers[id];
  }
  else {
    silentOTRecvers[id] = this;
    while(silentOTSenders[id] == nullptr);
    other = silentOTSenders[id];
  }

  nTree = numTrees;
  depth = logOT - log2((float) nTree) + 1;
  numOT = pow(2, logOT);
}

void SilentOT::baseOT_send() {
  std::vector<std::future<std::array<std::vector<GPUBlock>, 2>>> workers;
  for (int t = 0; t < nTree; t++) {
    workers.push_back(std::async([t, this]() {
      return SimplestOT(SimplestOT::Sender, t).send(depth+1);
    }));
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    leftHash.push_back(res[0]);
    rightHash.push_back(res[1]);
  }
}

void SilentOT::baseOT_recv() {
  std::vector<std::future<std::vector<GPUBlock>>> workers;
  for (int t = 0; t < nTree; t++) {
    workers.push_back(std::async([t, this]() {
      return SimplestOT(SimplestOT::Recver, t).recv(depth+1, rand());
    }));
  }
  leftHash.resize(nTree);
  rightHash.resize(nTree);
  for (int i = 0; i < nTree; i++) {
    leftHash.at(i).resize(depth+1);
    rightHash.at(i).resize(depth+1);
  }
  for (auto &worker : workers) {
    auto res = worker.get();
    choiceHash.push_back(res);
  }
}

std::pair<GPUBlock, GPUBlock> SilentOT::send() {
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;

  baseOT_send();
  auto [fullVector, delta] = pprf_send(root);
  GPUBlock fullVectorHashed(numOT * BLK_SIZE);
  return std::pair<GPUBlock, GPUBlock>(); //debug

  if (numOT < CHUNK_SIDE) {
    randMatrix = init_rand(prng, 2 * numOT, numOT);
    gen_rand(prng, randMatrix); // transposed
    hash_sender(fullVectorHashed, randMatrix, fullVector, 0);
  }
  else {
    randMatrix = init_rand(prng, CHUNK_SIDE, CHUNK_SIDE);
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        gen_rand(prng, randMatrix);
        hash_sender(fullVectorHashed, randMatrix, fullVector, chunkC);
      }
    }
  }
  del_rand(prng, randMatrix);
  return {fullVectorHashed, delta};
}

std::pair<GPUBlock, GPUBlock> SilentOT::recv() {
  baseOT_recv();
  auto [puncVector, choiceVector] = pprf_recv(choices);
  GPUBlock puncVectorHashed(numOT * BLK_SIZE);
  GPUBlock choiceVectorHashed(numOT * BLK_SIZE);
  return std::pair<GPUBlock, GPUBlock>(); //debug

  if (numOT < CHUNK_SIDE) {
    randMatrix = init_rand(prng, 2 * numOT, numOT);
    gen_rand(prng, randMatrix); // transposed
    hash_recver(puncVectorHashed, choiceVectorHashed, randMatrix, puncVector, choiceVector, 0, 0);
  }
  else {
    randMatrix = init_rand(prng, CHUNK_SIDE, CHUNK_SIDE);
    for (size_t chunkR = 0; chunkR < 2 * numOT / CHUNK_SIDE; chunkR++) {
      for (size_t chunkC = 0; chunkC < numOT / CHUNK_SIDE; chunkC++) {
        gen_rand(prng, randMatrix);
        hash_recver(puncVectorHashed, choiceVectorHashed, randMatrix, puncVector, choiceVector, chunkR, chunkC);
      }
    }
  }
  del_rand(prng, randMatrix);
  return {puncVectorHashed, choiceVectorHashed};
}

std::pair<GPUBlock, GPUBlock> SilentOT::pprf_send(TreeNode root) {
  size_t numLeaves = pow(2, depth);
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  EventLog::start(BufferInit);
  GPUBlock delta(BLK_SIZE);
  delta.clear();
  delta.set(123456);

  GPUBlock input(nTree * numLeaves * BLK_SIZE);
  GPUBlock output(nTree * numLeaves * BLK_SIZE);
  std::vector<GPUBlock> leftNodes(nTree, GPUBlock(numLeaves * BLK_SIZE / 2));
  std::vector<GPUBlock> rightNodes(nTree, GPUBlock(numLeaves * BLK_SIZE / 2));
  Aes aesLeft(k0_blk);
  Aes aesRight(k1_blk);

  for (int t = 0; t < nTree; t++) {
    output.set((uint8_t*) root.data, BLK_SIZE, t * numLeaves * BLK_SIZE);
  }
  EventLog::end(BufferInit);

  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    input = output;

    EventLog::start(PprfSenderExpand);
    for (int t = 0; t < nTree; t++) {
      TreeNode *inPtr = (TreeNode*) input.data_d + t * numLeaves;
      TreeNode *outPtr = (TreeNode*) output.data_d + t * numLeaves;
      aesLeft.expand_async(outPtr, leftNodes.at(t), inPtr, width, 0);
      aesRight.expand_async(outPtr, rightNodes.at(t), inPtr, width, 1);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfSenderExpand);

    EventLog::start(SumNodes);
    for (int t = 0; t < nTree; t++) {
      leftNodes.at(t).sum_async(BLK_SIZE);
      rightNodes.at(t).sum_async(BLK_SIZE);
    }
    cudaDeviceSynchronize();
    EventLog::end(SumNodes);

    EventLog::start(HashSender);
    for (int t = 0; t < nTree; t++) {
      other->leftHash.at(t).at(d-1) = leftHash.at(t).at(d-1) ^= leftNodes.at(t);
      other->rightHash.at(t).at(d-1) = rightHash.at(t).at(d-1) ^= rightNodes.at(t);
    }
    if (d == depth) {
      for (int t = 0; t < nTree; t++) {
        leftHash.at(t).at(d) ^= leftNodes.at(t);
        other->leftHash.at(t).at(d) = leftHash.at(t).at(d) ^= delta;
        rightHash.at(t).at(d) ^= rightNodes.at(t);
        other->rightHash.at(t).at(d) = rightHash.at(t).at(d) ^= delta;
      }
    }
    cudaDeviceSynchronize();
    other->msgDelivered++;
    EventLog::end(HashSender);
  }

  return {output, delta};
}

__global__
void set_choice(SparseVector choiceVector, int index, int t) {
  choiceVector.nonZeros[t] = index;
}

std::pair<GPUBlock, SparseVector> SilentOT::pprf_recv(uint64_t *choices) {
  size_t numLeaves = pow(2, depth);
  uint64_t k0 = 3242342, k1 = 8993849;
  uint8_t k0_blk[16] = {0};
  uint8_t k1_blk[16] = {0};

  memcpy(&k0_blk[8], &k0, sizeof(k0));
  memcpy(&k1_blk[8], &k1, sizeof(k1));

  EventLog::start(BufferInit);
  GPUBlock input(nTree * numLeaves * BLK_SIZE);
  GPUBlock output(nTree * numLeaves * BLK_SIZE);
  std::vector<GPUBlock> leftNodes(nTree, GPUBlock(numLeaves * BLK_SIZE / 2));
  std::vector<GPUBlock> rightNodes(nTree, GPUBlock(numLeaves * BLK_SIZE / 2));
  std::vector<SimplestOT*> baseOT;
  Aes aesLeft(k0_blk);
  Aes aesRight(k1_blk);
  std::vector<size_t> puncture(nTree, 0);

  SparseVector choiceVector = {
    .nBits = numLeaves,
  };
  cudaError_t err = cudaMalloc(&choiceVector.nonZeros, nTree * sizeof(size_t));
  if (err != cudaSuccess)
    fprintf(stderr, "choice vec: %s\n", cudaGetErrorString(err));
  EventLog::end(BufferInit);

  auto &sum = choiceHash; // alias
  for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
    input = output;

    EventLog::start(PprfRecverExpand);
    // expand layer
    for (int t = 0; t < nTree; t++) {
      TreeNode *inPtr = (TreeNode*) input.data_d + t * width;
      TreeNode *outPtr = (TreeNode*) output.data_d + t * width;
      aesLeft.expand_async(outPtr, leftNodes.at(t), inPtr, width, 0);
      aesRight.expand_async(outPtr, rightNodes.at(t), inPtr, width, 1);
    }
    cudaDeviceSynchronize();
    EventLog::end(PprfRecverExpand);

    while(msgDelivered < d);
    EventLog::start(HashRecver);
    // once left sum^hash and right sum^hash ready, unhash to obtain sum
    for (int t = 0; t < nTree; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      if (choice == 0)
        sum.at(t).at(d-1) ^= leftHash.at(t).at(d-1);
      else
        sum.at(t).at(d-1) ^= rightHash.at(t).at(d-1);

      if (d == depth) {
        if (choice == 0)
          sum.at(t).at(d) ^= rightHash.at(t).at(d);
        else
          sum.at(t).at(d) ^= leftHash.at(t).at(d);
      }
    }
    cudaDeviceSynchronize();

    // insert obtained sum into layer
    for (int t = 0; t < nTree; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      GPUBlock *side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      TreeNode *sideCasted = (TreeNode*) side->data_d;
      int recvNodeId = puncture.at(t) * 2 + choice;
      cudaMemcpy(&sideCasted[recvNodeId / 2], sum.at(t).at(d-1).data_d, BLK_SIZE, cudaMemcpyDeviceToDevice);

      if (d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        sideCasted = (TreeNode*) xorSide->data_d;
        size_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        cudaMemcpy(&sideCasted[deltaNodeId / 2], sum.at(t).at(d).data_d, BLK_SIZE, cudaMemcpyDeviceToDevice);
      }
    }
    EventLog::end(HashRecver);

    // conduct sum/xor in parallel
    EventLog::start(SumNodes);
    for (int t = 0; t < nTree; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      GPUBlock *side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      side->sum_async(BLK_SIZE);

      if (d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        xorSide->sum_async(BLK_SIZE);
      }
    }
    cudaDeviceSynchronize();
    EventLog::end(SumNodes);

    // insert active node obtained from sum into output
    for (int t = 0; t < nTree; t++) {
      int choice = (choices[t] & (1 << d-1)) >> d-1;
      GPUBlock *side = choice == 0 ? &leftNodes.at(t) : &rightNodes.at(t);
      TreeNode *oCasted = (TreeNode*) output.data_d + t * numLeaves;
      int recvNodeId = puncture.at(t) * 2 + choice;
      cudaMemcpy(&oCasted[recvNodeId], side->data_d, BLK_SIZE, cudaMemcpyDeviceToDevice);

      if(d == depth) {
        GPUBlock *xorSide = choice == 0 ? &rightNodes.at(t) : &leftNodes.at(t);
        size_t deltaNodeId = puncture.at(t) * 2 + (1-choice);
        cudaMemcpy(&oCasted[deltaNodeId], xorSide->data_d, BLK_SIZE, cudaMemcpyDeviceToDevice);
      }
    }
  }

  for (int t = 0; t < nTree; t++) {
    set_choice<<<1, 1>>>(choiceVector, t*numLeaves + puncture.at(t), t);
    cudaDeviceSynchronize();
    choiceVector.weight++;
  }

  return {output, choiceVector};
}
