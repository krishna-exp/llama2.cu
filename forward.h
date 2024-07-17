#ifndef FORWARD_H_INCLUDED
#define FORWARD_H_INCLUDED

#include "model.h"

static inline void expAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("assert failed: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define expErrChk(err) { expAssert((err), __FILE__, __LINE__); }


// CUDA accelerated functions
void rmsnorm_cu(float *sb, float* o, const float* x, const float* w, size_t size, int nBatches = BATCH_SIZE, cudaStream_t stream = 0);
void softmax_cu(float* sb, float* x, size_t size, size_t stride, int nBatches = BATCH_SIZE, cudaStream_t stream = 0);
void matmul_cu(float* xout, const float* x, const float* wT, size_t n, size_t d, int nBatches = BATCH_SIZE, cudaStream_t stream = 0);

void transpose_cu(float *odata, const float *idata, size_t w, size_t h, cudaStream_t stream = 0);

// The actual forward function
void setup(Transformer* transformer);

// expecting BATCH_SIZE # of tokens
// all of which will be processed in one shot
// check the logits in the transformer for outputs
void forward(Transformer* transformer, int* token, int nBatches, int pos);

#endif // FORWARD_H_INCLUDED
