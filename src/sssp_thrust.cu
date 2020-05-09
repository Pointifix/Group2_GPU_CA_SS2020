#include "sssp_thrust.h"

SSSP_Thrust::SSSP_Thrust(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}


__device__ int getGlobalIdx_3D_3DT(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void SSSP_Kernel1(const int* edges, const int* destinations, const int* weights, int* previous_node, int* mask,
                             const int* cost, int* update_cost, int nodes_amount, int edges_amount)
{
    int tid = getGlobalIdx_3D_3DT();

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

__global__ void SSSP_Kernel2(int* mask, int* cost, int* update_cost, int nodes_amount)
{
    int tid = getGlobalIdx_3D_3DT();

    if (tid >= nodes_amount) return;

    if(cost[tid] > update_cost[tid])
    {
        cost[tid] = update_cost[tid];
        mask[tid] = true;
    }

    update_cost[tid] = cost[tid];
}

std::vector<std::vector<int>> SSSP_Thrust::compute(int source_node)
{

    thrust::host_vector<int> previous_node(graph->edges.size(), -1);
    thrust::host_vector<int> mask(graph->edges.size(), 0);
    thrust::host_vector<int> cost(graph->edges.size(), std::numeric_limits<int>::max());
    thrust::host_vector<int> update_cost(graph->edges.size(), std::numeric_limits<int>::max());

    mask[source_node] = true;
    cost[source_node] = 0;
    update_cost[source_node] = 0;

    thrust::device_vector<int> d_edges = graph->edges;
    thrust::device_vector<int> d_destinations = graph->destinations;
    thrust::device_vector<int> d_weights = graph->weights;
    thrust::device_vector<int> d_previous_node = previous_node;
    thrust::device_vector<int> d_mask = mask;
    thrust::device_vector<int> d_cost = cost;
    thrust::device_vector<int> d_update_cost;

    // while we still find false in the mask (Ma not empty)
    while (std::find(mask.begin(), mask.end(), true) != mask.end())
    {
        int numBlocks = ceil((double)graph->edges.size() / 1024);

        dim3 threadsPerBlock(32, 32);
        M_CFUN((SSSP_Kernel1<<<numBlocks, threadsPerBlock>>>(
                thrust::raw_pointer_cast(&d_edges[0]), thrust::raw_pointer_cast(&d_destinations[0]),
                thrust::raw_pointer_cast(&d_weights[0]), thrust::raw_pointer_cast(&d_previous_node[0]),
                thrust::raw_pointer_cast(&d_mask[0]), thrust::raw_pointer_cast(&d_cost[0]),
                thrust::raw_pointer_cast(&d_update_cost[0]), graph->edges.size(), graph->destinations.size())));

        M_CFUN((SSSP_Kernel2<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(&d_mask[0]),
                thrust::raw_pointer_cast(&d_cost[0]), thrust::raw_pointer_cast(&d_update_cost[0]), graph->edges.size())));

        //copy back mask
        mask = d_mask;
    }

    // no need to clean up vectors as they get de-allocated when they go out of scope
    previous_node = d_previous_node;
    cost = d_cost;
    update_cost = d_update_cost;

    std::cout << "\n\nMask: ";
    for(int i = 0; i < mask.size(); i++)
    {
        std::cout << mask[i] << ",";
    }
    std::cout << "\n\nCost: ";
    for(int i = 0; i < cost.size(); i++)
    {
        std::cout << cost[i] << ",";
    }
    std::cout << "\n\nUpdateCost: ";
    for(int i = 0; i < update_cost.size(); i++)
    {
        std::cout << update_cost[i] << ",";
    }
    std::cout << "\n\nPreviousNode: ";
    for(int i = 0; i < previous_node.size(); i++)
    {
        std::cout << previous_node[i] << ",";
    }

    // TODO
    return std::vector<std::vector<int>>();
}