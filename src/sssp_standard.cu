#include "sssp_standard.h"

SSSP_Standard::SSSP_Standard(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

//https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
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

std::shared_ptr<Paths> SSSP_Standard::compute(int source_node)
{
    std::vector<int> previous_nodes(graph->edges.size(), -1);
    std::vector<int> mask(graph->edges.size(), 0);
    std::vector<int> cost(graph->edges.size(), std::numeric_limits<int>::max()), update_cost(graph->edges.size(),
            std::numeric_limits<int>::max());

    mask.at(source_node) = true;
    cost.at(source_node) = 0;
    update_cost.at(source_node) = 0;

    int *d_edges = nullptr;
    int *d_destinations = nullptr;
    int *d_weights = nullptr;
    int *d_previous_node = nullptr;
    int *d_mask = nullptr;
    int *d_cost = nullptr;
    int *d_update_cost = nullptr;

    M_C(cudaMalloc((void**) &d_edges,          graph->edges.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_destinations,   graph->destinations.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_weights,        graph->weights.size() * sizeof(int)));

    M_C(cudaMalloc((void**) &d_previous_node, previous_nodes.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_mask, mask.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_cost, cost.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_update_cost, update_cost.size() * sizeof(int)));

    M_C(cudaMemcpy(d_edges,        &graph->edges[0],        graph->edges.size() * sizeof(int),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_destinations, &graph->destinations[0], graph->destinations.size() * sizeof(int),   cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_weights,      &graph->weights[0],      graph->weights.size() * sizeof(int),        cudaMemcpyHostToDevice));

    M_C(cudaMemcpy(d_previous_node,&previous_nodes[0],  previous_nodes.size() * sizeof(int),cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_mask,         &mask[0],            mask.size() * sizeof(int),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_cost,         &cost[0],            cost.size() * sizeof(int),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_update_cost,  &update_cost[0],     update_cost.size() * sizeof(int),   cudaMemcpyHostToDevice));

    // while we still find false in the mask (Ma not empty)
    while (std::find(mask.begin(), mask.end(), true) != mask.end())
    {
        int numBlocks = ceil((double)graph->edges.size() / 1024);

        dim3 threadsPerBlock(32, 32);
        M_CFUN((CUDA_SSSP_Kernel1<<<numBlocks, threadsPerBlock>>>(d_edges, d_destinations, d_weights,
                d_previous_node, d_mask, d_cost, d_update_cost, graph->edges.size(), graph->destinations.size())));

        M_CFUN((CUDA_SSSP_Kernel2<<<numBlocks, threadsPerBlock>>>(d_mask, d_cost, d_update_cost, graph->edges.size())));

        //copy back mask
        M_C(cudaMemcpy(&mask[0], d_mask, mask.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }

    M_C(cudaMemcpy(&previous_nodes[0], d_previous_node, previous_nodes.size() * sizeof(int), cudaMemcpyDeviceToHost));
    M_C(cudaMemcpy(&cost[0], d_cost, cost.size() * sizeof(int), cudaMemcpyDeviceToHost));
    M_C(cudaMemcpy(&update_cost[0], d_update_cost, update_cost.size() * sizeof(int), cudaMemcpyDeviceToHost));

    M_C(cudaFree(d_edges));
    M_C(cudaFree(d_destinations));
    M_C(cudaFree(d_weights));
    M_C(cudaFree(d_previous_node));
    M_C(cudaFree(d_mask));
    M_C(cudaFree(d_cost));
    M_C(cudaFree(d_update_cost));

    std::shared_ptr<Paths> paths = std::make_shared<Paths>(Paths(previous_nodes, source_node, graph));

    return paths;
}