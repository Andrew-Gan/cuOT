#ifndef __MYTYPES_H__
#define __MYTYPES_H__

#define AES_BLOCKLEN 16
#define AES_KEYLEN 16 // Key length in bytes
#define AES_keyExpSize 176
#define NUM_ROUNDS 10
#define PADDED_LEN 1024

#define TREENODE_SIZE AES_BLOCKLEN
#define NUM_SAMPLES 16

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <wmmintrin.h>

typedef struct {
  uint8_t roundKey[320];
} AES_ctx;

typedef struct {
  size_t length;
  uint8_t *content;
} AES_buffer;

typedef struct {
  AES_ctx *ctx;
  AES_buffer *buf;
  size_t start;
  size_t end;
} ThreadArgs;

typedef struct {
  uint32_t data[TREENODE_SIZE / 4];
} TreeNode;

typedef struct {
  void (*encryptor)(AES_ctx*, AES_buffer*, int);
  TreeNode *tree;
  size_t idx;
} ThreadTreeArgs;

#include <bitset>
#include <pair>

class Matrix {
private:
    int row = 0;
    int col = 0;
    std::bitset<32*64> content;
public:
    Matrix(int row, int col) {
        // TODO: use row and col
        content.reset();
    }

    uint8_t get(int row, int col) {
        return content.test(row * col + col);
    }

    void set(int row, int col) {
        content.set(row * col + col);
    }

    void clear(int row, int col) {
        content.reset(row * col + col);
    }

    std::pair<int, int> getDim() {
        return std::make_pair(row, col);
    }

    void print() {
        std::string str;
        str << content;
        for (int i = 0; i < str.length() / col; i++) {
            std::cout << str.substr(i * col, );
        }
    }

    Matrix operator*(const Matrix& rhs) {
        Matrix res(row, rhs.col);
        for(int r = 0; r < row; r++) {
            for(int c = 0; c < rhs.col; c++) {
                for(int k = 0; k < col; k++) {
                    res.set(r, c) += content[r * col + k] * content[c + k * row];
                }
            }
        }
        return res;
    }
};

#endif
