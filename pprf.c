#include "aesni.h"
#include "aesgpu.h"
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

AES_ctx aesKeys[2];

void test_expand(AES_block *tree, size_t depth, void (*initialiser)(), void (*encryptor) (), const char *msg) {
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
    // This thread will process 8 trees at a time. It will interlace
    // the sets of trees are processed with the other threads.
    // For each level perform the following.

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    const int numSamples = 8;
    for(int i = 0; i < numSamples; i++) {
        int layerStartingIdx = 1;
        int width = 1;
        for (size_t d = 0; d < depth; d++) {
            width *= 2;

            // pad 1024 to reduce thread divergence in cuda implementation
            AES_block *leftChildren = malloc(sizeof(*tree) * (width / 2 + 1024));
            AES_block *rightChildren = malloc(sizeof(*tree) * (width / 2 + 1024));

            size_t leftID = 0, rightID = 0;
            for (size_t idx = layerStartingIdx; idx < layerStartingIdx + width; idx++) {
                size_t parentIdx = (idx - 1) / 2;
                // copy left children to array
                if (idx % 2) {
                    memcpy(&leftChildren[leftID++], &tree[parentIdx], sizeof(*tree));
                }
                // copy right children to array
                else {
                    memcpy(&rightChildren[rightID++], &tree[parentIdx], sizeof(*tree));
                }
            }

            // perform parallel aes hash on both arrays
            AES_buffer leftBuf = {
                .content = (uint8_t*) leftChildren,
                .length = sizeof(*tree) * width / 2,
            };
            AES_buffer rightBuf = {
                .content = (uint8_t*) rightChildren,
                .length = sizeof(*tree) * width / 2,
            };

            encryptor(&aesKeys[0], &leftBuf, 1);
            encryptor(&aesKeys[1], &rightBuf, 1);

            // copy back
            leftID = 0, rightID = 0;
            for (size_t idx = layerStartingIdx; idx < layerStartingIdx + width; idx++) {
                // copy left children to array
                if (idx % 2) {
                    memcpy(&tree[idx], &leftChildren[leftID++], sizeof(*tree));
                }
                // copy right children to array
                else {
                    memcpy(&tree[idx], &rightChildren[rightID++], sizeof(*tree));
                }
            }

            layerStartingIdx += width;

            free(leftChildren);
            free(rightChildren);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    float duration = (end.tv_sec - start.tv_sec) * 1000;
    duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("Tree expansion using %s: %0.4f ms\n", msg, duration / numSamples);
}
