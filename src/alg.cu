#include "alg.cuh"
#include "common.cuh"

namespace alg {

    void countoccur_seq(const std::vector<unsigned int> &a, std::vector<unsigned int> &out) {
        for (unsigned int v : a) {
            out[v]++;
        }
    }

    void prefixsum_seq(const std::vector<unsigned int> &a, std::vector<unsigned int> &out) {
        M_A(a.size() == out.size());
        if (a.empty()) return;

        out[0] = a[0];
        for (int i = 1; i < a.size(); i++) {
            out[i] = out[i-1] + a[i];
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    // PARALLEL IMPLEMENTATIONS (CUDA)
    // -----------------------------------------------------------------------------------------------------------------

    __global__ void _countoccur_parcu(const unsigned int *a, int na, unsigned int *out) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < na) {
            unsigned int val = a[i];
            atomicAdd(&out[val], 1u); // Increment
        }
    }
    void countoccur_parcu(const std::vector<unsigned int> &a, std::vector<unsigned int> &out) {
        if (a.empty()) return;
        if (out.empty()) return;

        unsigned int *d_a, *d_out;
        size_t sizea = a.size() * sizeof(unsigned int);
        size_t sizeout = out.size() * sizeof(unsigned int);

        int threadsPerBlock = 256;
        int numBlocks = (int) ceil((float)a.size() / (float)threadsPerBlock);

        M_C(cudaMalloc((void **) &d_a, sizea));
        M_C(cudaMalloc((void **) &d_out, sizeout));
        {
            M_C(cudaMemcpy(d_a, a.data(), sizea, cudaMemcpyHostToDevice));
            M_CFUN((_countoccur_parcu<<< numBlocks, threadsPerBlock >>>(d_a, a.size(), d_out)));
            M_C(cudaMemcpy(out.data(), d_out, sizeout, cudaMemcpyDeviceToHost));
        }
        M_C(cudaFree(d_a));
        M_C(cudaFree(d_out));
    }

    __global__ void _add_parcu(const unsigned int *a, const unsigned int *b, unsigned int *out, int n) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < n) {
            out[i] = a[i] + b[i];
        }
    }
    void add_parcu(const std::vector<unsigned int> &a,
                   const std::vector<unsigned int> &b,
                   std::vector<unsigned int> &out) {
        M_A(a.size() == b.size());
        M_A(a.size() == out.size());
        if (a.empty()) return;

        size_t N = a.size();
        const size_t size = N * sizeof(unsigned int);

        int threadsPerBlock = 256;
        int numBlocks = (int) ceil((float)N / (float)threadsPerBlock);

        unsigned int *d_a, *d_b, *d_out;

        M_C(cudaMalloc((void **) &d_a, size));
        M_C(cudaMalloc((void **) &d_b, size));
        M_C(cudaMalloc((void **) &d_out, size));
        {
            M_C(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
            M_C(cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice));
            M_CFUN((_add_parcu<<< numBlocks, threadsPerBlock >>>(d_a, d_b, d_out, N)));
            M_C(cudaMemcpy(out.data(), d_out, size, cudaMemcpyDeviceToHost));
        }
        M_C(cudaFree(d_a));
        M_C(cudaFree(d_b));
        M_C(cudaFree(d_out));
    }



}