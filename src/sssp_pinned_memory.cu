#include "sssp_pinned_memory.h"

#include <utility>


SSSP_Pinned_Memory::SSSP_Pinned_Memory(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::shared_ptr<Paths> SSSP_Pinned_Memory::compute(int source_node)
{
    int* previous_nodes = nullptr;
    mask_t* mask = nullptr;
    int* cost = nullptr;

    M_C(cudaMallocHost((int**) &previous_nodes, graph->edges.size() * sizeof(int)));
    M_C(cudaMallocHost((mask_t**) &mask, graph->edges.size() * sizeof(mask_t)));
    M_C(cudaMallocHost((int**) &cost, graph->edges.size() * sizeof(int)));

    //https://stackoverflow.com/questions/15947969/memset-an-int-16-bit-array-to-shorts-max-value
    //memset(previous_nodes, -1, graph->edges.size() * sizeof(int));
    memset(mask, M_MASK_FALSE, graph->edges.size() * sizeof(mask_t));
    //memset(cost, std::numeric_limits<int>::max(), graph->edges.size() * sizeof(int));

    for (int i = 0; i < graph->edges.size(); i++)
    {
        cost[i] = std::numeric_limits<int>::max();
        previous_nodes[i] = M_INVALID_POSITION;
    }

    mask[source_node] = M_MASK_TRUE;
    cost[source_node] = 0;

    int *d_edges = nullptr;
    int *d_destinations = nullptr;
    int *d_weights = nullptr;
    int *d_previous_node = nullptr;
    mask_t *d_mask = nullptr;
    int *d_cost = nullptr;

    M_C(cudaMalloc((void**) &d_edges,          graph->edges.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_destinations,   graph->destinations.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_weights,        graph->weights.size() * sizeof(int)));

    M_C(cudaMalloc((void**) &d_previous_node,   graph->edges.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_mask,            graph->edges.size() * sizeof(mask_t)));
    M_C(cudaMalloc((void**) &d_cost,            graph->edges.size() * sizeof(int)));

    M_C(cudaMemcpy(d_edges,        &graph->edges[0],        graph->edges.size() * sizeof(int),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_destinations, &graph->destinations[0], graph->destinations.size() * sizeof(int),   cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_weights,      &graph->weights[0],      graph->weights.size() * sizeof(int),        cudaMemcpyHostToDevice));

    M_C(cudaMemcpy(d_previous_node,previous_nodes,  graph->edges.size() * sizeof(int),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_mask,         mask,            graph->edges.size() * sizeof(mask_t),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_cost,         cost,            graph->edges.size() * sizeof(int),          cudaMemcpyHostToDevice));

    // while we still find false in the mask (Ma not empty)
    while (std::find(mask, mask + graph->edges.size(), true) != mask + graph->edges.size())
    {
        int numBlocks = ceil((double)graph->edges.size() / M_BLOCKSIZE);

        M_CFUN((alg::SSSP_Kernel<<<numBlocks, M_BLOCKSIZE>>>(d_edges, d_destinations, d_weights,
                d_previous_node, d_mask, d_cost, graph->edges.size(), graph->destinations.size())));

        //copy back mask
        M_C(cudaMemcpy(mask, d_mask, graph->edges.size() * sizeof(mask_t), cudaMemcpyDeviceToHost));
    }

    M_C(cudaMemcpy(previous_nodes, d_previous_node, graph->edges.size() * sizeof(int), cudaMemcpyDeviceToHost));
    M_C(cudaMemcpy(cost, d_cost, graph->edges.size() * sizeof(int), cudaMemcpyDeviceToHost));

    M_C(cudaFree(d_edges));
    M_C(cudaFree(d_destinations));
    M_C(cudaFree(d_weights));
    M_C(cudaFree(d_previous_node));
    M_C(cudaFree(d_mask));
    M_C(cudaFree(d_cost));

    std::vector<int> ret_previous_nodes(previous_nodes, previous_nodes + graph->edges.size());
    std::vector<int> ret_cost(cost, cost + graph->edges.size());
    std::shared_ptr<Paths> paths = std::make_shared<Paths>(Paths(ret_previous_nodes,ret_cost , source_node, graph));

    M_C(cudaFreeHost(mask));
    M_C(cudaFreeHost(previous_nodes));
    M_C(cudaFreeHost(cost));

    return paths;
}