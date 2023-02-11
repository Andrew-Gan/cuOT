#include "aesni.h"
#include "aesgpu.h"
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

AES_ctx aesKeys[2];

// root is heapified array

void test_expand(AES_block *tree, size_t depth, void (*initialiser)(), void (*encryptor) (), size_t nThread, const char *msg) {
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

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int currIndex = 1;
    int width = 1;
    for (size_t d = 0; d < depth; ++d) {
        // The previous level of the GGM tree.
        size_t prevLevel = d;

        // The next level of the GGM tree that we are populating.
        size_t currLevel = d+1;

        // The total number of children in this level.
        width *= 2;

        if (width < nThread) {
            // keep parallelising until the next layer
        }
        else if (width == nThread) {
            // stop spawning new threads
            // instruct current threads to expand to leaf node
        }

        // For each child, populate the child by expanding the parent.
        for (uint64_t childIdx = currIndex; childIdx < currIndex + width; childIdx++) {
            // Index of the parent in the previous level.
            uint64_t parentIdx = (childIdx - 1) / 2;

            memcpy(&tree[childIdx], &tree[parentIdx], sizeof(*tree));

            AES_buffer buf = {
                .content = (uint8_t*)&(tree[childIdx]),
                .length = sizeof(*tree),
            };
            encryptor(&aesKeys[childIdx % 2], &buf, nThread);
        }

        currIndex += width;
    }

    gettimeofday(&end, NULL);

    float duration = (end.tv_sec - start.tv_sec) * 1000;
    duration += (end.tv_usec - start.tv_usec) / 1000;
    printf("Tree expansion using %s: %0.4f ms\n", msg, duration);
}
