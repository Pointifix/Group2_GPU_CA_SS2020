#ifndef GROUP2_GPU_CA_SS2020_ALG_CUH
#define GROUP2_GPU_CA_SS2020_ALG_CUH

#include "common.cuh"

#include <vector>

using uint = unsigned int;

namespace alg {

    __global__ void SSSP_Kernel(const m_t* edges, const m_t* destinations, const m_t* weights,
                                m_t* previous_node, int* mask, m_t* cost,
                                size_t nodes_amount, size_t edges_amount);

    /**
     * @param a Vector that will be filled
     * @param firstValue Value that will be in a[0]
     * @param increment a[i] will have the value: firstValue + (i * increment)
     */
    void fill_parcu(m_t *d_a, size_t Na, m_t firstValue, m_t increment=0);

}

#endif //GROUP2_GPU_CA_SS2020_ALG_CUH
