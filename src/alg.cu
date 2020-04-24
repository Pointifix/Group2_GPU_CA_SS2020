#include "alg.cuh"
#include "common.cuh"

namespace alg {

    void countoccur_seq(const std::vector<uint> &a, std::vector<uint> &out) {
        for (uint v : a) {
            out[v]++;
        }
    }

    void exscan_seq(const std::vector<uint> &a, std::vector<uint> &out) {
        M_A(a.size() == out.size());
        if (a.empty()) return;

        out[0] = 0;
        for (int i = 1; i < a.size(); i++) {
            out[i] = out[i-1] + a[i-1];
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    // PARALLEL IMPLEMENTATIONS (CUDA)
    // -----------------------------------------------------------------------------------------------------------------

    __global__ void _countoccur_parcu(const uint *a, int Na, uint *out) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < Na) {
            uint val = a[i];
            atomicAdd(&out[val], 1u); // Increment
        }
    }
    void countoccur_parcu(const std::vector<uint> &a, std::vector<uint> &out) {
        if (a.empty()) return;
        if (out.empty()) return;

        uint *d_a, *d_out;
        size_t sizea = a.size() * sizeof(uint);
        size_t sizeout = out.size() * sizeof(uint);

        int threadsPerBlock = M_BLOCKSIZE;
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

    __global__ void _add_parcu(const uint *a, const uint *b, uint *out, int N) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < N) {
            out[i] = a[i] + b[i];
        }
    }
    void add_parcu(const std::vector<uint> &a,
                   const std::vector<uint> &b,
                   std::vector<uint> &out) {
        M_A(a.size() == b.size());
        M_A(a.size() == out.size());
        if (a.empty()) return;

        size_t N = a.size();
        const size_t size = N * sizeof(uint);

        int threadsPerBlock = M_BLOCKSIZE;
        int numBlocks = (int) ceil((float)N / (float)threadsPerBlock);

        uint *d_a, *d_b, *d_out;

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

    /**
     * Source: https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
     * @param a
     * @param out
     * @param N
     */
    __global__ void _exscan_parcu(const uint *a, uint *out, int N) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < N) {

            // Up-sweep (reduce)
            

        }
    }
    void exscan_parcu(const std::vector<uint> &a, std::vector<uint> &out) {
        M_A(a.size() == out.size());
        if (a.empty()) return;

        size_t N = a.size();
        const size_t size = N * sizeof(uint);

        int threadsPerBlock = M_BLOCKSIZE;
        int numBlocks = (int) ceil((float)N / (float)threadsPerBlock);


    }

}