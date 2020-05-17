#include "alg.cuh"

namespace alg {
    __global__ void SSSP_Kernel(const pos_t* edges, const pos_t* destinations, const weight_t * weights,
                                pos_t* previous_node, mask_t* mask, weight_t * cost,
                                size_t nodes_amount, size_t edges_amount)
    {
        uint tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid >= nodes_amount) return;

        if (mask[tid])
        {
            pos_t first = edges[tid];
            pos_t last = (tid + 1 < nodes_amount) ? edges[tid + 1] : edges_amount;

            mask[tid] = M_MASK_FALSE;

            for (pos_t i = first; i < last; i++)
            {
                pos_t nid = destinations[i];

                if(cost[nid] > cost[tid] + weights[i])
                {
                    weight_t new_cost = cost[tid] + weights[i];

                    atomicMin(&cost[nid], new_cost);

                    if (cost[nid] == new_cost)
                    {
                        previous_node[nid] = tid;
                        mask[nid] = M_MASK_TRUE;
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

    template<class T> void set_parcu(T *d_a, pos_t position, T value) {
        M_CFUN((_fill_parcu<T><<< 1, 1 >>>(&d_a[position], 1, value)));
    }

    // -----------------------------------------------------------------------------------------------------------------
    // TEMPLATES
    // -----------------------------------------------------------------------------------------------------------------

    template void fill_parcu(bool *d_a, size_t Na, bool value);
    template void fill_parcu(int *d_a, size_t Na, int value);

    template void set_parcu(bool *d_a, pos_t position, bool value);
    template void set_parcu(int *d_a, pos_t position, int value);
}