#ifndef GROUP2_GPU_CA_SS2020_ALG_CUH
#define GROUP2_GPU_CA_SS2020_ALG_CUH

#include "common.cuh"

#include <vector>

using uint = unsigned int;

namespace alg {

    __global__ void SSSP_Kernel(const pos_t* edges, const pos_t* destinations, const weight_t * weights,
                                pos_t *previous_node, mask_t *mask, weight_t *cost,
                                size_t nodes_amount, size_t edges_amount);

    /**
     * @param a Vector that will be filled
     * @param value Value that 'a' will be filled with
     */
    template<class T>
    void fill_parcu(T *d_a, const size_t &Na, const T &value);

    template<class T>
    void set_parcu(T *d_a, const pos_t &position, const T &value);

    template <class T>
    void contains_parcu(const T *d_a, const size_t &Na, const T &value, bool &out);

}

#endif //GROUP2_GPU_CA_SS2020_ALG_CUH
