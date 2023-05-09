// seed gen -> seed expansion -> matrix gen -> random matrix hashing -> cot

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <random>

#include "unit_test.h"
#include "protocols.h"

uint64_t* gen_choices(int numTrees) {
  uint64_t *choices = (uint64_t*) malloc(sizeof(uint64_t) * numTrees);
  for (int t = 0; t < numTrees; t++) {
    choices[t] = ((uint64_t) rand() << 32) | rand();
  }
  return choices;
}

#include "aes.h"

int main(int argc, char** argv) {
  // test_cuda();
  // test_rsa();
  // test_aes();
  // test_base_ot();

  if (argc < 4) {
    fprintf(stderr, "Usage: ./ot protocol depth trees\n");
    return EXIT_FAILURE;
  }

  int protocol = atoi(argv[1]);
  int userDepth = atoi(argv[2]);
  size_t numOT = pow(2, userDepth);
  // each node has 2^7 bits
  // num bits in final layer = 2 * OT, to be halved during encoding
  size_t actualDepth = userDepth - 7 + 1;
  int numTrees = atoi(argv[3]);
  TreeNode root;
  root.data[0] = 123456;
  root.data[1] = 7890123;

  printf("OTs: %lu, Trees: %d\n", numOT, numTrees);

  uint64_t *choices = gen_choices(numTrees);
  switch (protocol) {
    case 1: silentOT(root, choices, actualDepth, numTrees);
      break;
  }

  free(choices);
  return EXIT_SUCCESS;
}
