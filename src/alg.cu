#include "alg.cuh"
#include "common.cuh"

namespace alg {

    //https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
    __device__ int getGlobalIdx_3D_3D(){
        int blockId = blockIdx.x + blockIdx.y * gridDim.x
                      + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                       + (threadIdx.z * (blockDim.x * blockDim.y))
                       + (threadIdx.y * blockDim.x) + threadIdx.x;
        return threadId;
    }

    __global__ void SSSP_Kernel(const int* edges, const int* destinations, const int* weights, int* previous_node, int* mask,
                                const int* cost, int nodes_amount, int edges_amount)
    {
        int tid = getGlobalIdx_3D_3D();

        if (tid >= nodes_amount) return;

        if (mask[tid])
        {
            int first = edges[tid];
            int last = (tid + 1 < nodes_amount) ? edges[tid + 1] : edges_amount;

            mask[tid] = false;

            for (int i = first; i < last; i++)
            {
                int nid = destinations[i];

                if(cost[nid] > cost[tid] + weights[i])
                {
                    int new_cost = cost[tid] + weights[i];

                    atomicMin((int*)&cost[nid], new_cost);

                    if (cost[nid] == new_cost)
                    {
                        previous_node[nid] = tid;
                        mask[nid] = true;
                    }
                }
            }
        }
    }

    void countoccur_seq(const std::vector<uint> &a, std::vector<uint> &out) {
        for (uint v : a) {
            out[v]++;
        }
    }

    void exscan_seq(const std::vector<uint> &a, std::vector<uint> &out, int offset) {
        M_A(a.size() == out.size());
        if (a.empty()) return;

        out[offset] = 0;
        for (int i = offset+1; i < a.size(); i++) {
            out[i] = out[i-1] + a[i-1];
        }
    }

    void add_seq(uint *a, size_t N, int v) {
        for (int i = 0; i < N; i++) {
            a[i] += v;
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    // PARALLEL IMPLEMENTATIONS (CUDA)
    // -----------------------------------------------------------------------------------------------------------------

    __global__ void _fill_parcu(uint *a, int N, uint firstValue, int increment) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < N) {
            a[i] = firstValue + (i * increment);
        }
    }
    void fill_parcu(std::vector<uint> &a, uint firstValue, int increment) {
        if (a.empty()) return;
        M_A((int)firstValue + (a.size() - 1) * increment >= 0);

        uint *d_a;
        size_t size = a.size() * sizeof(uint);

        int threadsPerBlock = M_BLOCKSIZE;
        int numBlocks = (int) ceil((float)a.size() / (float)threadsPerBlock);

        M_C(cudaMalloc((void **) &d_a, size));
        {
            M_C(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
            M_CFUN((_fill_parcu<<< numBlocks, threadsPerBlock >>>(d_a, a.size(), firstValue, increment)));
            M_C(cudaMemcpy(a.data(), d_a, size, cudaMemcpyDeviceToHost));
        }
        M_C(cudaFree(d_a));
    }

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
     * @param N Must be a power of 2 (like 2, 4, 8, 16, 32, 64, 128, 256, ...)
     */
    __global__ void _exscan_parcu(const uint *a, uint *out, int N) {

        __shared__ int shared[M_BLOCKSIZE]; // Allocated on invocation

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int tid = threadIdx.x;
        int offset = 1;

        // Load input into shared memory
        shared[2 * tid]     = a[2 * i];
        shared[2 * tid + 1] = a[2 * i + 1];

        // Sweep up (reduce)
        for (int d = N>>1; d > 0; d >>= 1) {
            __syncthreads();
            if (tid < d) {
                int src = offset * (2 * tid + 1) - 1;
                int dst = offset * (2 * tid + 2) - 1;

                shared[dst] = shared[src] + shared[dst];
            }
            offset *= 2;
        }

        // Clear the last element
        if (tid == 0) shared[N-1] = 0;

        // Sweep down
        for (int d = 1; d < N; d *= 2) {
            offset >>= 1;
            __syncthreads();
            if (tid < d) {
                int src = offset * (2 * tid + 1) - 1;
                int dst = offset * (2 * tid + 2) - 1;

                int tmpsrc = shared[src];
                shared[src] = shared[dst];
                shared[dst] = tmpsrc + shared[dst];
            }
        }

        __syncthreads();

        // Write results to output
        out[2 * i]     = shared[2 * tid];
        out[2 * i + 1] = shared[2 * tid + 1];
    }
    void exscan_parcu(const std::vector<uint> &a, std::vector<uint> &out) {
        M_A(a.size() == out.size());
        if (a.empty()) return;

        size_t N = a.size();
        const size_t size = N * sizeof(uint);

        int lastBlockSize = (N % M_BLOCKSIZE);
        int lastParBlockSize =  pow(2, (int) floor(std::log2(lastBlockSize)));
        int lastSeqBlockSize = lastBlockSize - lastParBlockSize;

        int remainingThreads = N - lastBlockSize;
        int numBlocks = (int) ceil((float)remainingThreads / (float)M_BLOCKSIZE);

        uint *d_a, *d_out;

        M_C(cudaMalloc((void **) &d_a, size));
        M_C(cudaMalloc((void **) &d_out, size));
        {
            M_C(cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice));
            if (numBlocks > 0) {
                M_CFUN((_exscan_parcu<<< numBlocks, M_BLOCKSIZE >>>(d_a, d_out, remainingThreads)));
            }
            if (lastParBlockSize > 0) {
                int s = lastParBlockSize;
                int start = N - lastBlockSize;
                M_CFUN((_exscan_parcu<<< 1, s >>>(&d_a[start], &d_out[start], s)));
            }
            M_C(cudaMemcpy(out.data(), d_out, size, cudaMemcpyDeviceToHost));
            if (lastSeqBlockSize > 0) {
                int start = N - lastSeqBlockSize;
                exscan_seq(a, out, start);
                int last = start == 0 ? 0 : out[start - 1];
                if (last > 0) add_seq(&out[start], lastSeqBlockSize, last);
            }
        }
        M_C(cudaFree(d_a));
        M_C(cudaFree(d_out));
    }

}