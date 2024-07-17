#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "forward.h"

// llama.c reference functions
static void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

static void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// self reference functions
static void transpose(float *xout, float *x, size_t n, size_t d) {
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            xout[i + j * d] = x[i * n + j];
        }
    }
}

// helper functions
static void compare(const char* tag, float *a, float *b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float v0 = abs(a[i]);
        float v1 = abs(b[i]);

        float diff = abs(v0 - v1);
        float sum = (v0 + v1) + 1e-5;

        // diff is already negligible
        if (diff < 1e-4) {
            continue;
        }

        if ((diff / sum) > 1e-3) {
            printf("Error in (%s) at idx: %zu, %f != %f\n", tag, i, a[i], b[i]);
            exit(1);
        }
    }

    printf("No Error in (%s)\n", tag);
}

static float randf() {
    const int upper = 10;
    const int lower = -10;

    return ((rand() % (upper - lower + 1)) + lower) / 10.0;
}

static void randff(float *x, size_t size) {
    for (size_t i = 0; i < size; i++) {
        // printf("Setting %zu\n", i);
        x[i] = randf();
    }
}

int main() {
    int nBatches = 32;
    size_t size = 288;

    float *sb;
    expErrChk(cudaMallocManaged(&sb, sizeof(float) * ((SCRATCH_BUFFER_SIZE * nBatches) + nBatches)));

    // rmsnorm testing
    {
        float *dORef;
        float *dO;
        float *dX;
        float *dW;

        expErrChk(cudaMallocManaged(&dORef, sizeof(float) * size * nBatches));
        expErrChk(cudaMallocManaged(&dO, sizeof(float) * size * nBatches));
        expErrChk(cudaMallocManaged(&dX, sizeof(float) * size * nBatches));
        expErrChk(cudaMallocManaged(&dW, sizeof(float) * size));

        randff(dX, size * nBatches);
        randff(dW, size);

        for (int b = 0; b < nBatches; b++) {
            rmsnorm(&dORef[b * size], &dX[b * size], dW, size);
        }

        rmsnorm_cu(sb, dO, dX, dW, size, nBatches, /* default stream */ 0);
        expErrChk(cudaDeviceSynchronize());

        compare("rmsnorm", dORef, dO, size * nBatches);

        expErrChk(cudaFree(dW));
        expErrChk(cudaFree(dX));
        expErrChk(cudaFree(dO));
        expErrChk(cudaFree(dORef));
    }

    // softmax testing
    {
        float *dX;
        float *dXRef;

        expErrChk(cudaMallocManaged(&dX, sizeof(float) * size * nBatches));
        expErrChk(cudaMallocManaged(&dXRef, sizeof(float) * size * nBatches));

        randff(dX, size * nBatches);
        expErrChk(cudaMemcpy(dXRef, dX, sizeof(float) * size * nBatches, cudaMemcpyDeviceToDevice));

        for (int b = 0; b < nBatches; b++) {
            softmax(&dXRef[b * size], size);
        }

        softmax_cu(sb, dX, size, size, nBatches, /* default stream */ 0);
        expErrChk(cudaDeviceSynchronize());

        compare("softmax", dXRef, dX, size * nBatches);

        expErrChk(cudaFree(dXRef));
        expErrChk(cudaFree(dX));
    }

    // matmul testing
    {
        float *dORef;
        float *dO;
        float *dX;
        float *dW;
        float *dWTRef;
        float *dWT;

        size_t n = 1100;
        size_t d = 32000;

        expErrChk(cudaMallocManaged(&dX, sizeof(float) * n * nBatches));
        expErrChk(cudaMallocManaged(&dORef, sizeof(float) * d * nBatches));
        expErrChk(cudaMallocManaged(&dO, sizeof(float) * d * nBatches));
        expErrChk(cudaMallocManaged(&dW, sizeof(float) * d * n));
        expErrChk(cudaMallocManaged(&dWTRef, sizeof(float) * d * n));
        expErrChk(cudaMallocManaged(&dWT, sizeof(float) * d * n));

        randff(dX, n * nBatches);
        randff(dW, n * d);

        transpose(dWTRef, dW, n, d);
        transpose_cu(dWT, dW, n, d);
        expErrChk(cudaDeviceSynchronize());

        compare("transpose", dWTRef, dWT, n * d);

        for (int b = 0; b < nBatches; b++) {
            matmul(&dORef[b * d], &dX[b * n], dW, n, d);
        }

        printf("CUDA kernel begin\n");
        matmul_cu(dO, dX, dWT, n, d, nBatches, /* default stream */ 0);
        expErrChk(cudaDeviceSynchronize());
        printf("CUDA kernel complete\n");

        compare("matmul", dORef, dO, d * nBatches);

        expErrChk(cudaFree(dWT));
        expErrChk(cudaFree(dWTRef));
        expErrChk(cudaFree(dW));
        expErrChk(cudaFree(dO));
        expErrChk(cudaFree(dORef));
        expErrChk(cudaFree(dX));
    }

    // testing complete
    expErrChk(cudaFree(sb));
    return 0;
}
