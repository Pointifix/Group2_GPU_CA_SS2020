#ifndef GROUP2_GPU_CA_SS2020_COMMON_CUH
#define GROUP2_GPU_CA_SS2020_COMMON_CUH

#define M_BLOCKSIZE 256
using uint = unsigned int;

__device__ int getGlobalIdx_3D_3D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void CUDA_SSSP_Kernel1(const int* edges, const int* destinations, const int* weights, int* previous_node, int* mask,
                                  const int* cost, int* update_cost, int nodes_amount, int edges_amount)
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

            if(update_cost[nid] > cost[tid] + weights[i])
            {
                update_cost[nid] = cost[tid] + weights[i];
                previous_node[nid] = tid;
            }
        }
    }
}

__global__ void CUDA_SSSP_Kernel2(int* mask, int* cost, int* update_cost, int nodes_amount)
{
    int tid = getGlobalIdx_3D_3D();

    if (tid >= nodes_amount) return;

    if(cost[tid] > update_cost[tid])
    {
        cost[tid] = update_cost[tid];
        mask[tid] = true;
    }

    update_cost[tid] = cost[tid];
}

#ifdef DEBUG

#include <cstdio>
#include <stdexcept>
#include <string>

namespace debug {
    class OurAssertException : public std::runtime_error {
    public:
        explicit OurAssertException(const std::string& message) : std::runtime_error(message) {}
    };
}

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

// ASSERT
#define M_A(a) if(!(a)) { \
    fprintf(stderr, "DEBUG ASSERTION ERROR \"%s\" (%s, line %i)\n", #a, __FILE__, __LINE__); \
    std::terminate(); \
} while(0)

#else
#define M_C(a) a
#define M_CFUN(a) a
#define M_D(a)
#define M_A(a)
#endif

#endif //GROUP2_GPU_CA_SS2020_COMMON_CUH
