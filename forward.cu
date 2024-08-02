#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <stdexcept>

#include "forward.h"
#include "model.h"

/**
 * The synchronization streams that need manage the parallelization
 */
class ModelSyncStreams {
    public:
    cudaStream_t syncStream;
    cudaStream_t qStream;
    cudaStream_t kStream;
    cudaStream_t vStream;
    cudaStream_t w1Stream;
    cudaStream_t w3Stream;

    ModelSyncStreams() {
        expErrChk(cudaStreamCreate(&syncStream));
        expErrChk(cudaStreamCreate(&qStream));
        expErrChk(cudaStreamCreate(&kStream));
        expErrChk(cudaStreamCreate(&vStream));
        expErrChk(cudaStreamCreate(&w1Stream));
        expErrChk(cudaStreamCreate(&w3Stream));
    }

    ModelSyncStreams(ModelSyncStreams& other) {
        throw std::runtime_error("no copies!");
    }

    ModelSyncStreams(ModelSyncStreams&& other) {
        throw std::runtime_error("no moves!");
    }

    ~ModelSyncStreams() {
        expErrChk(cudaStreamDestroy(syncStream));
        expErrChk(cudaStreamDestroy(qStream));
        expErrChk(cudaStreamDestroy(kStream));
        expErrChk(cudaStreamDestroy(vStream));
        expErrChk(cudaStreamDestroy(w1Stream));
        expErrChk(cudaStreamDestroy(w3Stream));
    }
};

/**
 * The transpose cuda kernel
 */
template<size_t threads>
__global__ static void transpose(float *odata, const float *idata, size_t w, size_t h) {
    // ref: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

    __shared__ float tile[threads][threads + 1];

    int x = blockIdx.x * threads + threadIdx.x;
    int y = blockIdx.y * threads + threadIdx.y;

    if (x < w && y < h)
        tile[threadIdx.y][threadIdx.x] = idata[(y * w) + x];

    __syncthreads();

    x = blockIdx.y * threads + threadIdx.x;
    y = blockIdx.x * threads + threadIdx.y;

    if (x < h && y < w)
        odata[(y * h) + x] = tile[threadIdx.x][threadIdx.y];
}

/**
 * The transpose cuda kernel wrapper
 */
void transpose_cu(float *odata, const float *idata, size_t w, size_t h, cudaStream_t stream) {
    const int threads = 16;
    const int threadsX = threads;
    const int threadsY = threads;
    const int blocksX = (w + threadsX - 1) / threadsX;
    const int blocksY = (h + threadsY - 1) / threadsY;

    dim3 threadsD(threadsX, threadsY);
    dim3 blocksD(blocksX, blocksY);

    transpose<threads><<<blocksD, threadsD, 0, stream>>>(odata, idata, w, h);
    expErrChk(cudaGetLastError());
}

/**
 * Load the buffers using the stream 
 */
static void loadAhead(RunState* s, TransformerWeights* w, Config *p, int l, cudaStream_t stream) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int vocab_size = p->vocab_size;

    expErrChk(cudaMemcpyAsync(s->mb.d_raw, w->rms_att_weight + (l * dim), sizeof(float) * dim, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaMemcpyAsync(s->mb.d_wqT, w->wqT + l*dim*dim, sizeof(float) * dim * dim, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaMemcpyAsync(s->mb.d_wkT, w->wkT + l*dim*kv_dim, sizeof(float) * dim * kv_dim, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaMemcpyAsync(s->mb.d_wvT, w->wvT + l*dim*kv_dim, sizeof(float) * dim * kv_dim, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaMemcpyAsync(s->mb.d_woT, w->woT + l*dim*dim, sizeof(float) * dim * dim, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaMemcpyAsync(s->mb.d_rffw, w->rms_ffn_weight + l*dim, sizeof(float) * dim, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaMemcpyAsync(s->mb.d_w1T, w->w1T + l*dim*hidden_dim, sizeof(float) * dim * hidden_dim, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaMemcpyAsync(s->mb.d_w2T, w->w2T + l*dim*hidden_dim, sizeof(float) * dim * hidden_dim, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaMemcpyAsync(s->mb.d_w3T, w->w3T + l*dim*hidden_dim, sizeof(float) * dim * hidden_dim, cudaMemcpyHostToDevice, stream));
}

/**
 * Swap the compute buffers and memory buffers
 */
static void swapBuffers(RunState* s) {
    float *d_raw = s->mb.d_raw;
    s->mb.d_raw = s->cb.d_raw;
    s->cb.d_raw = d_raw;

    float *d_wqT = s->mb.d_wqT;
    s->mb.d_wqT = s->cb.d_wqT;
    s->cb.d_wqT = d_wqT;

    float *d_wkT = s->mb.d_wkT;
    s->mb.d_wkT = s->cb.d_wkT;
    s->cb.d_wkT = d_wkT;

    float *d_wvT = s->mb.d_wvT;
    s->mb.d_wvT = s->cb.d_wvT;
    s->cb.d_wvT = d_wvT;

    float *d_woT = s->mb.d_woT;
    s->mb.d_woT = s->cb.d_woT;
    s->cb.d_woT = d_woT;

    float *d_rffw = s->mb.d_rffw;
    s->mb.d_rffw = s->cb.d_rffw;
    s->cb.d_rffw = d_rffw;

    float *d_w1T = s->mb.d_w1T;
    s->mb.d_w1T = s->cb.d_w1T;
    s->cb.d_w1T = d_w1T;

    float *d_w2T = s->mb.d_w2T;
    s->mb.d_w2T = s->cb.d_w2T;
    s->cb.d_w2T = d_w2T;

    float *d_w3T = s->mb.d_w3T;
    s->mb.d_w3T = s->cb.d_w3T;
    s->cb.d_w3T = d_w3T;
}

/**
 * Perform rmsnorm sum of squares on the data (this is a reduction kernel)
 */
template<size_t blocks, size_t threads>
__global__ static void rmsnorm_ss(float *o, const float *x, size_t size) {

    // ref: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

    // batch
    int by = blockIdx.y;
    // block
    int bx = blockIdx.x;

    __shared__ float ss[threads];

    ss[threadIdx.x] = 0;

    size_t idx = threadIdx.x + (bx * threads);

    // using threads as template to perform
    // compile time optimizations
    // assert(threads == blockDim.x);

    for (size_t i = idx; i < size; i += blocks * threads) {
        float v = x[i + (by * size)];
        float sq = v * v;
        ss[threadIdx.x] += sq;
    }

    __syncthreads();

    // todo: this can also be done using wrap shfl, check that out
    // all the sums must be reduced to one value now
    for (size_t t = threads >> 1; t > 0; t = t >> 1) {
        if (threadIdx.x < t) {
            ss[threadIdx.x] = ss[threadIdx.x] + ss[threadIdx.x + t];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        o[(by * blocks) + bx] = ss[0];
    }
}

/**
 * Perform the normalization on the input data
 */
template<size_t blocks, size_t threads>
__global__ static void rmsnorm_norm(float *o, const float *x, const float *w, const float *ss, size_t size) {
    const int by = blockIdx.y;
    const size_t idx = threadIdx.x + (blockIdx.x * threads);

    const float ssb = ss[by];
    const size_t boff = (by * size);

    for (int i = idx; i < size; i += blocks * threads) {
        o[i + boff] = w[i] * x[i + boff] * ssb;
    }
}

/**
 * RMS Norm wrapper for the cuda kernels
 */
void rmsnorm_cu(float* dsb, float* o, const float* x, const float* w, size_t size, int nBatches, cudaStream_t stream) {
    const int blocks = 32;
    const int threads = 128;

    assert(blocks <= SCRATCH_BUFFER_SIZE);

    float sb[blocks * nBatches];

    dim3 threadsDim(threads);
    dim3 blocksDim(blocks, nBatches);

    rmsnorm_ss<blocks, threads><<<blocksDim, threadsDim, 0, stream>>>(dsb, x, size);
    expErrChk(cudaMemcpyAsync(sb, dsb, sizeof(float) * blocks * nBatches, cudaMemcpyDeviceToHost, stream));
    expErrChk(cudaStreamSynchronize(stream));

    // calculate sum of squares
    float ss[nBatches];

    for (int i = 0; i < nBatches; i++) {
        ss[i] = 0.0f;

        for (int j = 0; j < blocks; j++) {
            ss[i] += sb[(i * blocks) + j];
        }

        ss[i] /= size;
        // to avoid divide by 0
        ss[i] += 1e-5f;
        ss[i] = 1.0f / sqrtf(ss[i]);
    }

    expErrChk(cudaMemcpyAsync(dsb, ss, sizeof(float) * nBatches, cudaMemcpyHostToDevice, stream));
    expErrChk(cudaStreamSynchronize(stream));

    rmsnorm_norm<blocks, threads><<<blocksDim, threadsDim, 0, stream>>>(o, x, w, dsb, size);
}

/**
 * Find the max of the input for the softmax (purely for numerical stability purposes)
 */
template<size_t blocks, size_t threads>
__global__ static void softmax_max(float *o, const float *x, size_t size, size_t stride) {

    int by = blockIdx.y;
    int bx = blockIdx.x;

    __shared__ float smax[threads];

    smax[threadIdx.x] = -INFINITY;

    size_t idx = threadIdx.x + (bx * threads);

    // using threads as template to perform
    // compile time optimizations
    // assert(threads == blockDim.x);

    for (size_t i = idx; i < size; i += blocks * threads) {
        float v = x[(by * stride) + i];
        smax[threadIdx.x] = max(smax[threadIdx.x], v);
    }

    __syncthreads();

    // all the maxes must be reduced to one value now
    for (size_t t = threads >> 1; t > 0; t = t >> 1) {
        if (threadIdx.x < t) {
            smax[threadIdx.x] = max(smax[threadIdx.x], smax[threadIdx.x + t]);
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        o[(by * blocks) + bx] = smax[0];
    }
}

/**
 * Perform the e^x and sum the e^x values
 */
template<size_t blocks, size_t threads>
__global__ static void softmax_exp_sum(float *o, float *x, float *maxes, size_t size, size_t stride) {
    // o = Σ (exp(x - max))
    // x = exp(x - max)

    int by = blockIdx.y;
    int bx = blockIdx.x;

    float max = maxes[by];

    __shared__ float ssum[threads];

    ssum[threadIdx.x] = 0;

    size_t idx = threadIdx.x + (bx * threads);

    for (size_t i = idx; i < size; i += blocks * threads) {
        float v = x[(by * stride) + i];
        float e = expf(v - max);

        x[(by * stride) + i] = e;
        ssum[threadIdx.x] = e + ssum[threadIdx.x];
    }

    __syncthreads();

    // all the sums must be reduced to one value now
    for (size_t t = threads >> 1; t > 0; t = t >> 1) {
        if (threadIdx.x < t) {
            ssum[threadIdx.x] = ssum[threadIdx.x] + ssum[threadIdx.x + t];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        o[(by * blocks) + bx] = ssum[0];
    }
}

/**
 * Normalize the softmax output
 */
template<size_t blocks, size_t threads>
__global__ static void softmax_norm(float *x, const float* inv_sum, size_t size, size_t stride) {
    // batch
    const int by = blockIdx.y;
    // input block size
    const int bx = blockIdx.x;

    const size_t boff = by * stride;
    const size_t idx = threadIdx.x + (bx * threads);

    const float bis = inv_sum[by];

    for (int i = idx; i < size; i += blocks * threads) {
        x[boff + i] = x[boff + i] * bis;
    }
}

/**
 * CUDA kernel wrapper for the softmax
 */
void softmax_cu(float* dsb, float* x, size_t size, size_t stride, int nBatches, cudaStream_t stream) {
    const int blocks = 32;
    const int threads = 128;

    assert(blocks <= SCRATCH_BUFFER_SIZE);

    // mostly the current token being generated
    // so, ranges from (0 - seq length / context length)
    // printf("softmax size: %zu\n", size);

    // 2 options:
    // A)
    // 1. find the max (reduction kernel)
    // 2. exp and sum (linear and reduction kernels)
    // 3. normalize (linear kernel)
    //
    // B)
    // 1. exp (linear kernel)
    // 2. max and sum (reduction kernel) (max is optional, only for precision)
    // 3. normalize (linear kernel)
    //
    // I think B is more performant
    // But, loses in precision. That's why opting for A.

    // find max value (for numerical stability)

    // sb size MUST be more than blocks

    float sb[blocks * nBatches];

    dim3 blocksDim(blocks, nBatches);
    dim3 threadsDim(threads);

    softmax_max<blocks, threads><<<blocksDim, threadsDim, 0, stream>>>(dsb, x, size, stride);
    expErrChk(cudaMemcpyAsync(sb, dsb, sizeof(float) * blocks * nBatches, cudaMemcpyDeviceToHost, stream));
    expErrChk(cudaStreamSynchronize(stream));

    float maxes[nBatches];

    for (int i = 0; i < nBatches; i++) {
        maxes[i] = -INFINITY;

        for (int j = 0; j < blocks; j++) {
            if (sb[j + (i * blocks)] > maxes[i]) {
                maxes[i] = sb[j + (i * blocks)];
            }
        }
    }

    expErrChk(cudaMemcpyAsync(&dsb[blocks * nBatches], maxes, sizeof(float) * nBatches, cudaMemcpyHostToDevice, stream));

    softmax_exp_sum<blocks, threads><<<blocksDim, threadsDim, 0, stream>>>(dsb, x, &dsb[blocks * nBatches], size, stride);
    expErrChk(cudaMemcpyAsync(sb, dsb, sizeof(float) * blocks * nBatches, cudaMemcpyDeviceToHost, stream));
    expErrChk(cudaStreamSynchronize(stream));

    float sums[nBatches];

    for (int i = 0; i < nBatches; i++) {
        sums[i] = 0.0f;

        for (int j = 0; j < blocks; j++) {
            sums[i] += sb[j + (i * blocks)];
        }

        sums[i] = 1.0 / sums[i];
    }

    expErrChk(cudaMemcpyAsync(dsb, sums, sizeof(float) * nBatches, cudaMemcpyHostToDevice, stream));

    softmax_norm<blocks, threads><<<blocksDim, threadsDim, 0, stream>>>(x, dsb, size, stride);
}

/**
 * Perform the matmul on the data
 */
template<size_t threads, int max_batches>
__global__ static void matmul(float* xout, const float* x, const float* wT, size_t n, size_t d, int nBatches) {
    int bx = blockIdx.x;

    assert(threadIdx.x < threads);

    size_t idx = threadIdx.x + (bx * threads);

    float val[max_batches];

    for (int b = 0; b < nBatches; b++) {
        val[b] = 0;
    }

    for (size_t jo = 0; jo < n; jo += threads) {
        __shared__ float xs[max_batches][threads];

        for (int b = 0; b < nBatches; b++) {
            if (jo + threadIdx.x < n) {
                xs[b][threadIdx.x] = x[(b * n) + jo + threadIdx.x];
            } else {
                xs[b][threadIdx.x] = 0;
            }
        }

        __syncthreads();

        if (idx < d) {
            for (size_t j = jo; j < min(jo + threads, n); j++) {
                float w = wT[idx + (j * d)];

                for (int b = 0; b < nBatches; b++) {
                    float xx = xs[b][j - jo];
                    val[b] = val[b] + (w * xx);
                }
            }
        }

        __syncthreads();
    }

    /*
    if (idx < d) {
        for (size_t j = 0; j < n; j++) {
            float w = wT[idx + (j * d)];

            for (int b = 0; b < nBatches; b++) {
                float xx = x[(b * n) + j];
                val[b] = val[b] + (w * xx);
            }
        }
    }
    */

    if (idx < d) {
        for (int b = 0; b < nBatches; b++) {
            xout[(b * d) + idx] = val[b];
        }
    }
}

/**
 * Wrapper on the matmul cuda kernel
 *
 * NOTE: The input weights needs to be transposed, this is needed
 *       for an efficient kernel.
 *
 * NOTE: This kernel is not half as performant as the device peak.
 *       A much more performant kernel is in the sibling repository.
 */
void matmul_cu(float* xout, const float* x, const float* wT, size_t n, size_t d, int nBatches, cudaStream_t stream) {
    // weights must've been transposed

    // printf("matmul: dim: n: %zu, d: %zu\n", n, d);

    assert(nBatches <= BATCH_SIZE);

    const size_t threads = 128;
    size_t blocks = (d + threads - 1) / threads;

    dim3 blocksDim(blocks);
    dim3 threadsDim(threads);

    matmul<threads, BATCH_SIZE><<<blocksDim, threadsDim, 0, stream>>>(xout, x, wT, n, d, nBatches);
}

/**
 * Add two vectors (for feed forward)
 */
template<size_t blocks, size_t threads>
__global__ static void vadd(float* x, float* y, size_t size) {
    size_t idx = threadIdx.x + (blockIdx.x * threads);

    for (int i = idx; i < size; i += blocks * threads) {
        x[i] = x[i] + y[i];
    }
}

/**
 * Wrapper for the vadd cuda kernel
 */
void vadd_cu(float* x, float* y, size_t size, int nBatches, cudaStream_t stream) {
    const int blocks = 32;
    const int threads = 256;

    vadd<blocks, threads><<<blocks, threads, 0, stream>>>(x, y, size * nBatches);
}

/**
 * Perform the activation swiglu on the data
 */
__global__ static void swiglu(float *x, const float *w, size_t size) {
    size_t idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (idx < size) {
        float in = x[idx];
        float sc = w[idx];
        float d = 1.0f / (1.0f + expf(-in));

        x[idx] = in * sc * d;
    }
}

/**
 * Wrapper on the swiglu cuda kernel
 */
void swiglu_cu(float *x, const float *w, size_t size, int nBatches, cudaStream_t stream) {
    size_t linearSize = size * nBatches;

    const int threads = 128;
    size_t blocks = (linearSize + threads - 1) / threads;

    dim3 blocksDim(blocks);
    dim3 threadsDim(threads);

    swiglu<<<blocksDim, threadsDim, 0, stream>>>(x, w, linearSize);
}

/**
 * Perform the Rotational position encoding
 */
__global__ static void rope(float *k, float *q, int pos, int dim, int kv_dim, int head_size) {
    int b = blockIdx.y;
    size_t t_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    size_t qoff = b * dim;
    size_t koff = b * kv_dim;

    // 2 elements needs to be processed by a thread
    int idx = t_idx * 2;
    int head_dim = idx % head_size;

    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);

    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    if (idx < dim) {
        float v0 = q[qoff + idx];
        float v1 = q[qoff + idx + 1];
        q[qoff + idx] = v0 * fcr - v1 * fci;
        q[qoff + idx + 1] = v0 * fci + v1 * fcr;
    }

    if (idx < kv_dim) {
        float v0 = k[koff + idx];
        float v1 = k[koff + idx + 1];
        k[koff + idx] = v0 * fcr - v1 * fci;
        k[koff + idx + 1] = v0 * fci + v1 * fcr;
    }
}

/**
 * CUDA kernel wrapper on the rope
 */
static void rope_cu(float *k, float *q, int pos, int dim, int kv_dim, int head_size, int nBatches, cudaStream_t stream) {
    const int threads = 64;
    const int blocks = (dim + threads - 1) / threads;

    dim3 threadsDim(threads);
    dim3 blocksDim(blocks, nBatches);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    rope<<<blocksDim, threadsDim, 0, stream>>>(k, q, pos, dim, kv_dim, head_size);
}

/**
 * Print tensor for debugging
 */
static inline void print_tensor(const char* tag, const float *data, size_t size, int nBatches, cudaStream_t stream) {
    float buf[size * nBatches];

    cudaMemcpyAsync(buf, data, sizeof(float) * size * nBatches, cudaMemcpyDeviceToHost, stream);
    expErrChk(cudaStreamSynchronize(stream));

    printf("%s:\n", tag);

    for (int b = 0; b < nBatches; b++) {
        printf("batch[%d]: \n", b);
        for (size_t s = 0; s < size; s++) {
            printf("%f, ", buf[s + (b * size)]);
        }
        printf("\n");
    }
}

/**
 * The first half of attention.
 * Finding the q & k
 *
 * Not half as good as flash attention. But works.
 */
__global__ static void attention_x(float *att, const float *q, const float *kc, float inv_sqrt_dk,
                                    int pos, int kv_dim, int head_size, int seq_len, int loff, int kv_mul) {

    int batch = blockIdx.z;
    int n_batches = gridDim.z;

    int head = blockIdx.y;
    int n_heads = gridDim.y;

    int t = threadIdx.x + (blockIdx.x * blockDim.x);

    int b_att_off = batch * n_heads * seq_len;
    int h_att_off = head * seq_len;
    int att_off = b_att_off + h_att_off;

    int b_q_off = batch * n_heads * head_size;
    int h_q_off = head * head_size;
    int q_off = b_q_off + h_q_off;

    int b_kc_off = batch * (n_heads / kv_mul) * head_size;
    int h_kc_off = (head / kv_mul) * head_size;
    int kc_off = loff + b_kc_off + h_kc_off;

    if (t <= pos) {
        float score = 0.0f;

        for (int i = 0; i < head_size; i++) {
            score += q[q_off + i] * kc[kc_off + (t * kv_dim * n_batches) + i];
        }

        score *= inv_sqrt_dk;

        // save the score to the attention buffer
        att[att_off + t] = score;
    }
}

/**
 * The second half of attention
 * Finding the v
 */
__global__ static void attention_y(float *o, const float* vc, const float *att,
                                   int pos, int kv_dim, int head_size, int seq_len, int loff, int kv_mul) {

    int batch = blockIdx.z;
    int n_batches = gridDim.z;

    int head = blockIdx.y;
    int n_heads = gridDim.y;

    int t = threadIdx.x + (blockIdx.x * blockDim.x);

    int b_att_off = batch * n_heads * seq_len;
    int h_att_off = head * seq_len;
    int att_off = b_att_off + h_att_off;

    int b_o_off = batch * n_heads * head_size;
    int h_o_off = head * head_size;
    int o_off = b_o_off + h_o_off;

    int b_vc_off = batch * (n_heads / kv_mul) * head_size;
    int h_vc_off = (head / kv_mul) * head_size;
    int vc_off = loff + b_vc_off + h_vc_off;

    if (t <= pos) {
        const float a = att[att_off + t];

        for (int i = 0; i < head_size; i++) {
            atomicAdd(&o[o_off + i], a * vc[vc_off + (t * kv_dim * n_batches) + i]);
        }
    }
}

/**
 * Run the attention on the input data (for all the batches)
 */
static inline void attention(RunState* s, Config* p, int pos, int loff, int nBatches, cudaStream_t stream) {
    // this is parallelized across heads and batches

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int vocab_size = p->vocab_size;
    int n_heads = p->n_heads;
    int seq_len = p->seq_len;

    const float inv_sqrt_dk = 1.0 / sqrtf(head_size);

    // llama2
    // head_size = 128
    // n_heads   = 32

    // multihead attention.
    const int threads = 128;
    const size_t blocksX = (pos + threads) / threads;
    const size_t blocksY = n_heads;
    const size_t blocksZ = nBatches;

    dim3 blocksDim(blocksX, blocksY, blocksZ);
    dim3 threadsDim(threads);

    // attention to tokens including the current one. So, (pos + threads) instead of (pos + threads - 1)
    attention_x<<<blocksDim, threadsDim, 0, stream>>>(s->att, s->q, s->key_cache, inv_sqrt_dk, pos, kv_dim, head_size, seq_len, loff, kv_mul);

    // print_tensor("att run", s->att, seq_len * n_heads, nBatches, stream);

    softmax_cu(s->d_sb, s->att, pos + 1, seq_len, nBatches * n_heads, stream);

    attention_y<<<blocksDim, threadsDim, 0, stream>>>(s->xb, s->value_cache, s->att, pos, kv_dim, head_size, seq_len, loff, kv_mul);
}

/**
 * Forward pass on the input tokens
 *
 * Algorithm:
 * Load the weights of the first layer, wait for the load
 * For each layer in the network, perform the computation while loading for the next layer
 * Wait for both the computation and load to complete
 * Swap the compute and memory buffers
 * Finally after all the layers complete computation, complete the matmul with classifier weights
 */
void forward(Transformer* transformer, int* tokens, int nBatches, int pos) {
    // printf("forward\n");

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int vocab_size = p->vocab_size;

    // create the streams
    cudaStream_t loadStream;
    ModelSyncStreams batchStreams;

    expErrChk(cudaStreamCreate(&loadStream));

    // printf("forward starting\n");

    // load for the first layer
    loadAhead(s, w, p, 0, loadStream);
    expErrChk(cudaStreamSynchronize(loadStream));

    // copy the token embedding into x
    // todo: maybe copy once to device buffers and copy it n-1 times
    for (int b = 0; b < nBatches; b++) {
        float* content_row = w->token_embedding_table + (tokens[b] * dim);
        expErrChk(cudaMemcpyAsync(&s->x[b * dim], content_row, dim * sizeof(float), cudaMemcpyHostToDevice, batchStreams.syncStream));
    }

    expErrChk(cudaStreamSynchronize(batchStreams.syncStream));

    // print_tensor("input x", s->x, dim, nBatches, batchStreams.syncStream);

    // forward all the layers
    for (size_t l = 0; l < p->n_layers; l++) {
        swapBuffers(s);

        // copy the weights to all the layers
        if (l + 1 < p->n_layers) {
            loadAhead(s, w, p, l + 1, loadStream);
        }

        // key and value point to the kv cache
        int loff = l * p->seq_len * nBatches * kv_dim;

        s->k = s->key_cache + loff + pos * nBatches * kv_dim;
        s->v = s->value_cache + loff + pos * nBatches * kv_dim;

        // attention rmsnorm
        rmsnorm_cu(s->d_sb, s->xb, s->x, s->cb.d_raw, dim, nBatches, batchStreams.syncStream);
        expErrChk(cudaStreamSynchronize(batchStreams.syncStream));

        // qkv matmuls for this position
        matmul_cu(s->q, s->xb, s->cb.d_wqT, dim, dim, nBatches, batchStreams.qStream);
        matmul_cu(s->k, s->xb, s->cb.d_wkT, dim, kv_dim, nBatches, batchStreams.kStream);
        matmul_cu(s->v, s->xb, s->cb.d_wvT, dim, kv_dim, nBatches, batchStreams.vStream);

        expErrChk(cudaStreamSynchronize(batchStreams.qStream));
        expErrChk(cudaStreamSynchronize(batchStreams.kStream));
        expErrChk(cudaStreamSynchronize(batchStreams.vStream));

        expErrChk(cudaMemsetAsync(s->xb, 0, sizeof(float) * dim * nBatches, batchStreams.syncStream));
        expErrChk(cudaStreamSynchronize(batchStreams.syncStream));

        rope_cu(s->k, s->q, pos, dim, kv_dim, head_size, nBatches, batchStreams.syncStream);

        attention(s, p, pos, loff, nBatches, batchStreams.syncStream);

        // print_tensor("q", s->q, dim, nBatches, batchStreams.syncStream);
        // print_tensor("k", s->k, kv_dim, nBatches, batchStreams.syncStream);
        // print_tensor("v", s->v, kv_dim, nBatches, batchStreams.syncStream);
        // print_tensor("xb", s->xb, dim, nBatches, batchStreams.syncStream);
        // print_tensor("att", s->att, p->seq_len * p->n_heads, nBatches, batchStreams.syncStream);

        // final matmul to get the output of the attention
        matmul_cu(s->xb2, s->xb, s->cb.d_woT, dim, dim, nBatches, batchStreams.syncStream);

        // residual connection back into x
        vadd_cu(s->x, s->xb2, dim, nBatches, batchStreams.syncStream);

        // ffn rmsnorm
        rmsnorm_cu(s->d_sb, s->xb, s->x, s->cb.d_rffw, dim, nBatches, batchStreams.syncStream);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul_cu(s->hb, s->xb, s->cb.d_w1T, dim, hidden_dim, nBatches, batchStreams.w1Stream);
        matmul_cu(s->hb2, s->xb, s->cb.d_w3T, dim, hidden_dim, nBatches, batchStreams.w3Stream);

        expErrChk(cudaStreamSynchronize(batchStreams.w1Stream));
        expErrChk(cudaStreamSynchronize(batchStreams.w3Stream));
        swiglu_cu(s->hb, s->hb2, hidden_dim, nBatches, batchStreams.syncStream);

        // final matmul to get the output of the ffn
        matmul_cu(s->xb, s->hb, s->cb.d_w2T, hidden_dim, dim, nBatches, batchStreams.syncStream);

        // residual connection
        vadd_cu(s->x, s->xb, dim, nBatches, batchStreams.syncStream);

        expErrChk(cudaStreamSynchronize(batchStreams.syncStream));
        expErrChk(cudaStreamSynchronize(loadStream));

        // printf("Layer %zu out:\n", l);
        // print_tensor("x", s->x, dim, nBatches, batchStreams.syncStream);
    }

    // final rmsnorm
    rmsnorm_cu(s->d_sb, s->x, s->x, s->d_rfw, dim, nBatches, batchStreams.syncStream);

    // classifier into logits
    matmul_cu(s->logits, s->x, s->d_wclsT, dim, vocab_size, nBatches, batchStreams.syncStream);

    // wait for computation to complete
    expErrChk(cudaStreamSynchronize(batchStreams.syncStream));

    // destroy the streams
    // batchStreams are automatically destroyed
    expErrChk(cudaStreamDestroy(loadStream));

    // check the logits in the RunState
    // you confused caller.
}

/**
 * Bootstrap before forwarding anything
 */
void setup(Transformer* transformer) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    int dim = p->dim;
    int vocab_size = p->vocab_size;

    float *sb;
    expErrChk(cudaMalloc(&sb, sizeof(float) * vocab_size * dim));

    expErrChk(cudaMemcpy(s->d_rfw, w->rms_final_weight, sizeof(float) * dim, cudaMemcpyHostToDevice));

    expErrChk(cudaMemcpy(sb, w->wcls, sizeof(float) * dim * vocab_size, cudaMemcpyHostToDevice));
    transpose_cu(s->d_wclsT, sb, dim, vocab_size);

    expErrChk(cudaFree(sb));
}
