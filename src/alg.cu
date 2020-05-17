#include "alg.cuh"

namespace alg {
    __global__ void SSSP_Kernel(const data_t* edges, const data_t* destinations, const data_t* weights,
                                data_t* previous_node, bool* mask, data_t* cost,
                                size_t nodes_amount, size_t edges_amount)
    {
        uint tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid >= nodes_amount) return;

        if (mask[tid])
        {
            data_t first = edges[tid];
            data_t last = (tid + 1 < nodes_amount) ? edges[tid + 1] : edges_amount;

            mask[tid] = false;

            for (data_t i = first; i < last; i++)
            {
                data_t nid = destinations[i];

                if(cost[nid] > cost[tid] + weights[i])
                {
                    data_t new_cost = cost[tid] + weights[i];

                    atomicMin(&cost[nid], new_cost);

                    if (cost[nid] == new_cost)
                    {
                        previous_node[nid] = tid;
                        mask[nid] = true;
                    }

                }
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    // PARALLEL IMPLEMENTATIONS (CUDA)
    // -----------------------------------------------------------------------------------------------------------------

    template<class T> __global__ void _fill_parcu(T *a, size_t N, T value) {
        uint i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < N) {
            a[i] = value;
        }
    }

    template<class T> void fill_parcu(T *d_a, size_t Na, T value) {
        int threadsPerBlock = M_BLOCKSIZE;
        int numBlocks = (int) ceil((float)Na / (float)threadsPerBlock);
        M_CFUN((_fill_parcu<<< numBlocks, threadsPerBlock >>>(d_a, Na, value)));
    }

    template<class T> void set_parcu(T *d_a, size_t position, T value) {
        M_CFUN((_fill_parcu<T><<< 1, 1 >>>(&d_a[position], 1, value)));
    }

    template void fill_parcu(mask_t *d_a, size_t Na, mask_t value);
    template void fill_parcu(data_t *d_a, size_t Na, data_t value);

    template void set_parcu(mask_t *d_a, size_t position, mask_t value);
    template void set_parcu(data_t *d_a, size_t position, data_t value);
}