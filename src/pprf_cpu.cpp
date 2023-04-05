#include "mytypes.h"
#include <cstring>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>

// OT content
const uint32_t choices[8] = {
  0b01111110011011100010000000111011,
  0b00101011101100101010011001110010,
  0b10110000110000100001110011100100,
  0b00100110101111000000011111011101,
  0b11001000111100000001000111010100,
  0b00111010001111010100011110110101,
  0b11001000111010111100110101100101,
  0b10100001111101000000110011000000,
};
static std::atomic<TreeNode*> prf;
static std::atomic<TreeNode*> otNodes;
static std::atomic<bool*> treeExpanded;

static void xor_prf(TreeNode *sum, std::atomic<TreeNode*>& prf, TreeNode *pprf, int start, int end) {
  for (int idx = start; idx <= end; idx++) {
    for (int i = 0; i < TREENODE_SIZE / 4; i++) {
      if (pprf != nullptr) {
        sum[idx].data[i] ^= prf[idx].data[i] ^ pprf[idx].data[i];
      }
      else {
        sum[idx].data[i] ^= prf[idx].data[i];
      }
    }
  }
}

void pprf_sender_cpu(TreeNode *root, size_t depth,
  void (*initialiser)(AES_ctx*, const uint8_t*),
  void (*encryptor) (AES_ctx*, AES_buffer*, int),
  int numTrees) {
  treeExpanded = (bool*) malloc(numTrees * sizeof(*treeExpanded));
  memset((void*) treeExpanded, (int) false, numTrees);
  size_t numLeaves = pow(2, depth);

  AES_ctx aesKeys[2];

  uint64_t k0 = 3242342;
  uint8_t k0_blk[16];
  memset(k0_blk, 0, sizeof(k0_blk));
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  initialiser(&aesKeys[0], k0_blk);

  uint64_t k1 = 8993849;
  uint8_t k1_blk[16];
  memset(k1_blk, 0, sizeof(k1_blk));
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  initialiser(&aesKeys[1], k1_blk);

  otNodes = (TreeNode*) malloc(sizeof(*otNodes) * depth);
  prf = (TreeNode*) malloc(sizeof(*prf) * (numLeaves + 1024));
  TreeNode *leftNodes = (TreeNode*) malloc(sizeof(*leftNodes) * (numLeaves / 2 + 1024));
  TreeNode *rightNodes = (TreeNode*) malloc(sizeof(*rightNodes) * (numLeaves / 2 + 1024));
  TreeNode *sumLeaves = (TreeNode*) malloc(sizeof(*prf) * (numLeaves + 1024));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for(int t = 0; t < numTrees; t++) {
    int puncturedIndex = 0;
    memcpy(prf, root, sizeof(*root));

    for (size_t d = 1, width = 2; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      memcpy(leftNodes, prf, sizeof(*prf) * width / 2);
      memcpy(rightNodes, prf, sizeof(*prf) * width / 2);

      // perform parallel aes hash on both arrays
      AES_buffer leftBuf = {
        .length = sizeof(*prf) * width / 2,
        .content = (uint8_t*) leftNodes,
      };
      AES_buffer rightBuf = {
        .length = sizeof(*prf) * width / 2,
        .content = (uint8_t*) rightNodes,
      };
      encryptor(&aesKeys[0], &leftBuf, 8);
      encryptor(&aesKeys[1], &rightBuf, 8);

      // copy back to correct position
      for (int idx = 0; idx < width; idx++) {
        if (idx % 2 == 0) {
          memcpy(&prf[idx], &leftNodes[idx/2], sizeof(*prf));
        }
        else {
          memcpy(&prf[idx], &rightNodes[(idx-1)/2], sizeof(*prf));
        }
      }

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncturedIndex * 2 + 1 - (width - 1) + choice;
      memcpy(&otNodes[d-1], &prf[otLeafLayerIdx], sizeof(*prf));
      puncturedIndex = puncturedIndex * 2 + 1 + (1 - choice);
    }

    treeExpanded[t] = true;
    int numThread = numLeaves < 8 ? numLeaves : 8;
    int workload = numLeaves / numThread;
    std::vector<std::thread> accumulator;
    for (int i = 0; i < numThread; i++) {
      int start = i * workload;
      int end = start + workload - 1;
      accumulator.push_back(std::thread(xor_prf, sumLeaves, std::ref(prf), nullptr, start, end));
    }
    for (int i = 0; i < numThread; i++) {
      accumulator.at(i).join();
    }
    while(treeExpanded[t] == true);
  }

  free(leftNodes);
  free(rightNodes);

  clock_gettime(CLOCK_MONOTONIC, &end);
  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree expansion sender using AESNI: %0.4f ms\n", duration / NUM_SAMPLES);
}

void pprf_recver_cpu(void (*initialiser)(AES_ctx*, const uint8_t*),
  void (*encryptor) (AES_ctx*, AES_buffer*, int),
  TreeNode *sparseVec, int *nonZeroRows, size_t depth, int numTrees) {

  size_t numLeaves = pow(2, depth);

  AES_ctx aesKeys[2];

  uint64_t k0 = 3242342;
  uint8_t k0_blk[16];
  memset(k0_blk, 0, sizeof(k0_blk));
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  initialiser(&aesKeys[0], k0_blk);

  uint64_t k1 = 8993849;
  uint8_t k1_blk[16];
  memset(k1_blk, 0, sizeof(k1_blk));
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  initialiser(&aesKeys[1], k1_blk);

  while(treeExpanded == nullptr);

  TreeNode *pprf = (TreeNode*) malloc(sizeof(*pprf) * (numLeaves + 1024));
  TreeNode *leftNodes = (TreeNode*) malloc(sizeof(*leftNodes) * (numLeaves / 2 + 1024));
  TreeNode *rightNodes = (TreeNode*) malloc(sizeof(*rightNodes) * (numLeaves / 2 + 1024));
  TreeNode *sumLeaves = (TreeNode*) malloc(sizeof(*sumLeaves) * (numLeaves + 1024));
  memset(sparseVec, 0, sizeof(*sparseVec) * numLeaves);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for(int t = 0; t < numTrees; t++) {
    while (!treeExpanded[t]);
    int choice = choices[t] & 1;
    int puncturedIndex = 2 - choice;
    memcpy(&pprf[choice], &otNodes[0], sizeof(*otNodes));

    for (size_t d = 2, width = 4; d <= depth; d++, width *= 2) {
      // copy previous layer for expansion
      memcpy(leftNodes, pprf, sizeof(*pprf) * width / 2);
      memcpy(rightNodes, pprf, sizeof(*pprf) * width / 2);

      // perform parallel aes hash on both arrays
      AES_buffer leftBuf = {
        .length = sizeof(*pprf) * width / 2,
        .content = (uint8_t*) leftNodes,
      };
      AES_buffer rightBuf = {
        .length = sizeof(*pprf) * width / 2,
        .content = (uint8_t*) rightNodes,
      };
      encryptor(&aesKeys[0], &leftBuf, 8);
      encryptor(&aesKeys[1], &rightBuf, 8);

      // copy back to correct position
      int idx = 0;
      for (int idx = 0; idx < width; idx++) {
        if (idx % 2 == 0) {
          memcpy(&pprf[idx], &leftNodes[idx/2], sizeof(*prf));
        }
        else {
          memcpy(&pprf[idx], &rightNodes[(idx-1)/2], sizeof(*prf));
        }
      }

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncturedIndex * 2 + 1 - (width - 1) + choice;
      memcpy(&pprf[otLeafLayerIdx], &otNodes[d-1], sizeof(*otNodes));
      puncturedIndex = puncturedIndex * 2 + 1 + (1 - choice);
    }

    int numThread = numLeaves < 8 ? numLeaves : 8;
    int workload = numLeaves / numThread;
    std::vector<std::thread> accumulator;
    for (int i = 0; i < numThread; i++) {
      int start = i * workload;
      int end = start + workload - 1;
      accumulator.push_back(std::thread(xor_prf, sparseVec, std::ref(prf), pprf, start, end));
    }
    for (int i = 0; i < numThread; i++) {
      accumulator.at(i).join();
    }
    nonZeroRows[t] = puncturedIndex;
    treeExpanded[t] = false;
  }

  free(leftNodes);
  free(rightNodes);

  clock_gettime(CLOCK_MONOTONIC, &end);
  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree expansion recver using AESNI: %0.4f ms\n", duration / NUM_SAMPLES);
}
