#include <cstring>
#include <cmath>
#include <thread>
#include <vector>
#include <atomic>

#include "mytypes.h"
#include "aesni.h"
#include "pprf_cpu.h"

// OT content
static std::atomic<TreeNode*> otNodes;
static std::atomic<bool*> treeExpanded;

static void xor_prf(TreeNode *sum, TreeNode *operand, int start, int end) {
  for (int idx = start; idx <= end; idx++) {
    for (int i = 0; i < TREENODE_SIZE / 4; i++) {
      sum[idx].data[i] ^= operand[idx].data[i];
    }
  }
}

std::pair<TreeNode*, uint64_t> pprf_sender_cpu(uint64_t *choices, TreeNode root, int depth, int numTrees) {

  treeExpanded = (bool*) malloc(numTrees * sizeof(*treeExpanded));
  memset((void*) treeExpanded, (int) false, numTrees);
  size_t numLeaves = pow(2, depth);

  AES_ctx aesKeys[2];

  uint64_t k0 = 3242342;
  uint8_t k0_blk[16];
  memset(k0_blk, 0, sizeof(k0_blk));
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  aesni_init_ctx(&aesKeys[0], k0_blk);

  uint64_t k1 = 8993849;
  uint8_t k1_blk[16];
  memset(k1_blk, 0, sizeof(k1_blk));
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  aesni_init_ctx(&aesKeys[1], k1_blk);

  uint64_t delta = 0;

  TreeNode* prf = (TreeNode*) malloc(sizeof(*prf) * (numLeaves + 1024));
  otNodes = (TreeNode*) malloc(sizeof(*otNodes) * depth);
  TreeNode *leftNodes = (TreeNode*) malloc(sizeof(*leftNodes) * (numLeaves / 2 + 1024));
  TreeNode *rightNodes = (TreeNode*) malloc(sizeof(*rightNodes) * (numLeaves / 2 + 1024));
  TreeNode *fullVector = (TreeNode*) malloc(sizeof(*prf) * (numLeaves + 1024));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for(int t = 0; t < numTrees; t++) {
    int puncIndex = 0;
    memcpy(prf, &root, sizeof(root));

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
      aesni_ecb_encrypt(&aesKeys[0], &leftBuf, 8);
      aesni_ecb_encrypt(&aesKeys[1], &rightBuf, 8);

      // copy back to correct position
      for (int idx = 0; idx < width; idx++) {
        if (idx % 2 == 0)
          memcpy(&prf[idx], &leftNodes[idx/2], sizeof(*prf));
        else
          memcpy(&prf[idx], &rightNodes[(idx-1)/2], sizeof(*prf));
      }

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncIndex * 2 + 1 - (width - 1) + choice;
      memcpy(&otNodes[d-1], &prf[otLeafLayerIdx], sizeof(*prf));
      puncIndex = puncIndex * 2 + 1 + (1 - choice);
    }

    treeExpanded[t] = true;
    int numThread = numLeaves < 8 ? numLeaves : 8;
    int workload = numLeaves / numThread;
    std::vector<std::thread> accumulator;
    for (int i = 0; i < numThread; i++) {
      int start = i * workload;
      int end = start + workload - 1;
      accumulator.push_back(std::thread(xor_prf, fullVector, prf, start, end));
    }
    for (int i = 0; i < numThread; i++) {
      accumulator.at(i).join();
    }
    while(treeExpanded[t] == true);
  }

  free(leftNodes);
  free(rightNodes);
  free(otNodes);
  free(prf);

  clock_gettime(CLOCK_MONOTONIC, &end);
  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree exp AESNI sender: %0.4f ms\n", duration / NUM_SAMPLES);

  return {fullVector, delta};
}

std::pair<TreeNode*, int*> pprf_recver_cpu(uint64_t *choices, int depth, int numTrees) {

  size_t numLeaves = pow(2, depth);

  AES_ctx aesKeys[2];

  uint64_t k0 = 3242342;
  uint8_t k0_blk[16];
  memset(k0_blk, 0, sizeof(k0_blk));
  memcpy(&k0_blk[8], &k0, sizeof(k0));
  aesni_init_ctx(&aesKeys[0], k0_blk);

  uint64_t k1 = 8993849;
  uint8_t k1_blk[16];
  memset(k1_blk, 0, sizeof(k1_blk));
  memcpy(&k1_blk[8], &k1, sizeof(k1));
  aesni_init_ctx(&aesKeys[1], k1_blk);

  while(treeExpanded == nullptr);

  TreeNode *pprf = (TreeNode*) malloc(sizeof(*pprf) * (numLeaves + 1024));
  TreeNode *leftNodes = (TreeNode*) malloc(sizeof(*leftNodes) * (numLeaves / 2 + 1024));
  TreeNode *rightNodes = (TreeNode*) malloc(sizeof(*rightNodes) * (numLeaves / 2 + 1024));
  TreeNode *puncVec = (TreeNode*) malloc(numLeaves * sizeof(*puncVec));
  memset(puncVec, 0, sizeof(*puncVec) * numLeaves);
  int *puncIndices = (int*) malloc(numTrees * sizeof(*puncIndices));

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for(int t = 0; t < numTrees; t++) {
    while (!treeExpanded[t]);
    int choice = choices[t] & 1;
    int puncIndex = 2 - choice;
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
      aesni_ecb_encrypt(&aesKeys[0], &leftBuf, 8);
      aesni_ecb_encrypt(&aesKeys[1], &rightBuf, 8);

      // copy back to correct position
      int idx = 0;
      for (int idx = 0; idx < width; idx++) {
        if (idx % 2 == 0) {
          memcpy(&pprf[idx], &leftNodes[idx / 2], sizeof(*pprf));
        }
        else {
          memcpy(&pprf[idx], &rightNodes[(idx-1) / 2], sizeof(*pprf));
        }
      }

      int choice = (choices[t] & (1 << d-1)) >> d-1;
      int otLeafLayerIdx = puncIndex * 2 + 1 - (width - 1) + choice;
      memcpy(&pprf[otLeafLayerIdx], &otNodes[d-1], sizeof(*otNodes));
      puncIndex = puncIndex * 2 + 1 + (1 - choice);
    }

    int numThread = numLeaves < 8 ? numLeaves : 8;
    int workload = numLeaves / numThread;
    std::vector<std::thread> accumulator;
    for (int i = 0; i < numThread; i++) {
      int start = i * workload;
      int end = start + workload - 1;
      accumulator.push_back(std::thread(xor_prf, puncVec, pprf, start, end));
    }
    for (int i = 0; i < numThread; i++) {
      accumulator.at(i).join();
    }
    puncIndices[t] = puncIndex;
    treeExpanded[t] = false;
  }

  free(leftNodes);
  free(rightNodes);

  clock_gettime(CLOCK_MONOTONIC, &end);
  float duration = (end.tv_sec - start.tv_sec) * 1000;
  duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
  printf("Tree exp AESNI recver: %0.4f ms\n", duration / NUM_SAMPLES);

  return {puncVec, puncIndices};
}
