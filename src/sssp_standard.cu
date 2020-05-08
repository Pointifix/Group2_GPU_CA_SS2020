#include "sssp_standard.h"

SSSP_Standard::SSSP_Standard(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

__device__ int getGlobalIdx_3D_3D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void CUDA_SSSP_Kernel1(int* edges, int* destinations, int* weights, int* mask, int* cost, int* update_cost, int nodes_amount, int edges_amount)
{
    int tid = getGlobalIdx_3D_3D();

    /*
    mask[tid] = 2 * tid;
    cost[tid] = 3 * tid;
    update_cost[tid] = tid;

    return;
     */

    if (mask[tid])
    {
        int first = edges[tid];
        int last = (tid + 1 < nodes_amount) ? edges[tid + 1] : edges_amount;

        mask[tid] = false;

        for (int i = first; i < last; i++)
        {
            int nid = destinations[i];

            if(update_cost[nid] > cost[tid] + weights[nid])
            {
                update_cost[nid] = cost[tid] + weights[nid];
            }
        }
    }
}

__global__ void CUDA_SSSP_Kernel2(int* mask, int* cost, int* update_cost)
{
    int tid = getGlobalIdx_3D_3D();

    if(cost[tid] > update_cost[tid])
    {
        cost[tid] = update_cost[tid];
        mask[tid] = true;
    }

    update_cost[tid] = cost[tid];
}

std::vector<std::vector<int>> SSSP_Standard::compute(int source_node)
{

    source_node = 0;
    int *h_edges = new int[6]{0, 2, 4, 5, 6, 7};
    int *h_destinations = new int[7]{1, 2, 2, 3, 4, 5, 3};
    int *h_weights = new int[7]{4, 2, 5, 10, 3, 11, 4};
    int *h_mask = new int[6]{1, 0, 0, 0, 0, 0};
    int *h_cost =  new int[6]{100, 100, 100, 100, 100, 100};
    int *h_update_cost = new int[6]{100, 100, 100, 100, 100, 100};

    int *d_edges = 0;
    int *d_destinations = 0;
    int *d_weights = 0;
    int *d_mask = 0;
    int *d_cost = 0;
    int *d_update_cost = 0;

    cudaError_t cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**) &d_edges,          6 * sizeof(int));
    cudaStatus = cudaMalloc((void**) &d_destinations,   7 * sizeof(int));
    cudaStatus = cudaMalloc((void**) &d_weights,        7 * sizeof(int));

    cudaStatus = cudaMalloc((void**) &d_mask, 6 * sizeof(int));
    cudaStatus = cudaMalloc((void**) &d_cost, 6 * sizeof(int));

    cudaStatus = cudaMalloc((void**) &d_update_cost, 6 * sizeof(int));

    cudaStatus = cudaMemcpy(d_edges,        h_edges,        6 * sizeof(int),          cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_destinations, h_destinations, 7 * sizeof(int),   cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_weights,      h_weights,      7 * sizeof(int),        cudaMemcpyHostToDevice);

    cudaStatus = cudaMemcpy(d_mask,         h_mask,            6 * sizeof(int),          cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_cost,         h_cost,            6 * sizeof(int),          cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(d_update_cost,  h_update_cost,     6 * sizeof(int),   cudaMemcpyHostToDevice);

    // while we still find false in the mask (Ma not empty)
    while (true)
    {
        // for each vertex V do in parallel
        for (int i = 0; i < graph->edges.size(); i++)
        {
            int numBlocks = ceil((double)graph->edges.size() / 1024);

            dim3 threadsPerBlock(32, 32);
            CUDA_SSSP_Kernel1<<<numBlocks, threadsPerBlock>>>(d_edges, d_destinations, d_weights,
                    d_mask, d_cost, d_update_cost, graph->edges.size(), graph->destinations.size());

            cudaDeviceSynchronize();
            break;

            CUDA_SSSP_Kernel2<<<numBlocks, threadsPerBlock>>>(d_mask, d_cost, d_update_cost);

            cudaDeviceSynchronize();

        }
        // wait for all kernels to finish, so we get the finalized mask back
        cudaDeviceSynchronize();

        //copy back mask
        cudaError_t err = cudaMemcpy(h_mask, d_mask, 6 * sizeof(int), cudaMemcpyDeviceToHost);
        break;
    }

    cudaMemcpy(h_cost, d_cost, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_update_cost, d_update_cost, 6 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\n\nMask: ";
    for(int i = 0; i < 6; i++)
    {
        std::cout << h_mask[i] << ",";
    }
    std::cout << "\n\nCost: ";
    for(int i = 0; i < 6; i++)
    {
        std::cout << h_cost[i] << ",";
    }
    std::cout << "\n\nUpdateCost: ";
    for(int i = 0; i < 6; i++)
    {
        std::cout << h_update_cost[i] << ",";
    }

    return std::vector<std::vector<int>>();
}