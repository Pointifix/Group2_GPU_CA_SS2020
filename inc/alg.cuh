#ifndef GROUP2_GPU_CA_SS2020_ALG_CUH
#define GROUP2_GPU_CA_SS2020_ALG_CUH

#include <vector>

namespace alg {

    /**
     * Element-wise addition of the elements of 'a' and 'b'. Result is saved into 'out'
     */
    void add_parcu(const std::vector<unsigned int> &a,
                   const std::vector<unsigned int> &b,
                   std::vector<unsigned int> &out);

    /**
     * @param a The occurrence of the elements of this array will be counted
     * @param out Result will be written to this vector. Its size must be equal to the number of distinct elements in a
     */
    void countoccur_seq(const std::vector<unsigned int> &a, std::vector<unsigned int> &out);
    void countoccur_parcu(const std::vector<unsigned int> &a, std::vector<unsigned int> &out);

    /**
     * Aka "prefix sum", aka "cumulative sum" (cumsum)
     * @param a This array's elements will be summed cumulatively
     * @param out This array will store the cumulative sum of 'a'
     */
    void exscan_seq(const std::vector<unsigned int> &a, std::vector<unsigned int> &out);
    void exscan_parcu(const std::vector<unsigned int> &a, std::vector<unsigned int> &out);

}

#endif //GROUP2_GPU_CA_SS2020_ALG_CUH
