#ifndef GROUP2_GPU_CA_SS2020_ALG_CUH
#define GROUP2_GPU_CA_SS2020_ALG_CUH

#include <vector>

using uint = unsigned int;

namespace alg {

    /**
     * @param a Vector that will be filled
     * @param firstValue Value that will be in a[0]
     * @param increment a[i] will have the value: firstValue + (i * increment)
     */
    void fill_parcu(std::vector<uint> &a, uint firstValue, int increment=0);

    /**
     * Element-wise addition of the elements of 'a' and 'b'. Result is saved into 'out'
     */
    void add_seq(uint *a, size_t N, int v);
    void add_parcu(const std::vector<uint> &a,
                   const std::vector<uint> &b,
                   std::vector<uint> &out);

    /**
     * @param a The occurrence of the elements of this array will be counted
     * @param out Result will be written to this vector. Its size must be equal to the number of distinct elements in a
     */
    void countoccur_seq(const std::vector<uint> &a, std::vector<uint> &out);
    void countoccur_parcu(const std::vector<uint> &a, std::vector<uint> &out);

    /**
     * Aka "prefix sum", aka "cumulative sum" (cumsum)
     * @param a This array's elements will be summed cumulatively
     * @param out This array will store the cumulative sum of 'a'
     */
    void exscan_seq(const std::vector<uint> &a, std::vector<uint> &out, int offset=0);
    void exscan_parcu(const std::vector<uint> &a, std::vector<uint> &out);

}

#endif //GROUP2_GPU_CA_SS2020_ALG_CUH
