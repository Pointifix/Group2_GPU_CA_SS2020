#include "alg.cuh"

namespace alg {
    __global__ void SSSP_Kernel(const m_t* edges, const m_t* destinations, const m_t* weights,
                                m_t* previous_node, int* mask, m_t* cost,
                                size_t nodes_amount, size_t edges_amount)
    {
        uint tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid >= nodes_amount) return;

        if (mask[tid])
        {
            m_t first = edges[tid];
            m_t last = (tid + 1 < nodes_amount) ? edges[tid + 1] : edges_amount;

            mask[tid] = false;

            for (m_t i = first; i < last; i++)
            {
                m_t nid = destinations[i];

                if(cost[nid] > cost[tid] + weights[i])
                {
                    m_t new_cost = cost[tid] + weights[i];

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

    __global__ void _fill_parcu(m_t *a, size_t N, m_t firstValue, m_t increment) {
        uint i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < N) {
            a[i] = firstValue + (i * increment);
        }
    }
    void fill_parcu(m_t *d_a, size_t Na, m_t firstValue, m_t increment) {
        int threadsPerBlock = M_BLOCKSIZE;
        int numBlocks = (int) ceil((float)Na / (float)threadsPerBlock);
        M_CFUN((_fill_parcu<<< numBlocks, threadsPerBlock >>>(d_a, Na, firstValue, increment)));
    }

}