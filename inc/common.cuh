#ifndef GROUP2_GPU_CA_SS2020_COMMON_CUH
#define GROUP2_GPU_CA_SS2020_COMMON_CUH

#define M_BLOCKSIZE 256
using uint = unsigned int;
using mask_t = bool;
using data_t = int;
static constexpr data_t M_INVALID = -1;

#ifdef DEBUG

#include <cstdio>
#include <stdexcept>
#include <string>

#define _M_CHECK_PRIVATE(a,b) { \
    cudaError_t error = a; \
    if(error != cudaSuccess) { \
        fprintf(stderr,"CUDA ERROR \"%s\" ON %s (%s, line %i)\n", cudaGetErrorString(error), #b, __FILE__, __LINE__); \
        /*exit(-1);*/ \
    } \
} while(0)

// CHECK
#define M_C(a) _M_CHECK_PRIVATE(a,a)

// CHECK FUNCTION
#define M_CFUN(a) { \
    a; \
    M_C(cudaDeviceSynchronize()); \
    _M_CHECK_PRIVATE(cudaGetLastError(), a); \
} while(0)

// DEBUG
#define M_D(a) a

// DEBUG + CHECK
#define M_DC(a) M_C(a)

// ASSERT
#define M_A(a) if(!(a)) { \
    fprintf(stderr, "DEBUG ASSERTION ERROR \"%s\" (%s, line %i)\n", #a, __FILE__, __LINE__); \
    std::terminate(); \
} while(0)

#else
#define M_C(a) a
#define M_CFUN(a) a
#define M_D(a)
#define M_DC(a)
#define M_A(a)
#endif

#endif //GROUP2_GPU_CA_SS2020_COMMON_CUH
