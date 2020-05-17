#ifndef GROUP2_GPU_CA_SS2020_ALG_CUH
#define GROUP2_GPU_CA_SS2020_ALG_CUH

#include "common.cuh"

#include <vector>

using uint = unsigned int;

namespace alg {

    __global__ void SSSP_Kernel(const data_t* edges, const data_t* destinations, const data_t* weights,
                                data_t* previous_node, bool* mask, data_t* cost,
                                size_t nodes_amount, size_t edges_amount);

    /**
     * @param a Vector that will be filled
     * @param value Value that 'a' will be filled with
     */
    template<class T>
    void fill_parcu(T *d_a, size_t Na, T value);

    template<class T>
    void set_parcu(T *d_a, size_t position, T value);

}

#endif //GROUP2_GPU_CA_SS2020_ALG_CUH
