#include "sssp_thrust.h"

SSSP_Thrust::SSSP_Thrust(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::shared_ptr<Paths> SSSP_Thrust::compute(int source_node)
{

    thrust::host_vector<int> previous_nodes(graph->edges.size(), -1);
    thrust::host_vector<int> mask(graph->edges.size(), 0);
    thrust::host_vector<int> cost(graph->edges.size(), std::numeric_limits<int>::max());
    thrust::host_vector<int> update_cost(graph->edges.size(), std::numeric_limits<int>::max());

    mask[source_node] = true;
    cost[source_node] = 0;
    update_cost[source_node] = 0;

    thrust::device_vector<int> d_edges = graph->edges;
    thrust::device_vector<int> d_destinations = graph->destinations;
    thrust::device_vector<int> d_weights = graph->weights;
    thrust::device_vector<int> d_previous_node = previous_nodes;
    thrust::device_vector<int> d_mask = mask;
    thrust::device_vector<int> d_cost = cost;
    thrust::device_vector<int> d_update_cost;

    // while we still find false in the mask (Ma not empty)
    while (std::find(mask.begin(), mask.end(), true) != mask.end())
    {
        int numBlocks = ceil((double)graph->edges.size() / 256);

        dim3 threadsPerBlock(256);
        M_CFUN((CUDA_SSSP_Kernel1<<<numBlocks, threadsPerBlock>>>(
                thrust::raw_pointer_cast(&d_edges[0]), thrust::raw_pointer_cast(&d_destinations[0]),
                thrust::raw_pointer_cast(&d_weights[0]), thrust::raw_pointer_cast(&d_previous_node[0]),
                thrust::raw_pointer_cast(&d_mask[0]), thrust::raw_pointer_cast(&d_cost[0]),
                thrust::raw_pointer_cast(&d_update_cost[0]), graph->edges.size(), graph->destinations.size())));

        M_CFUN((CUDA_SSSP_Kernel2<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(&d_mask[0]),
                thrust::raw_pointer_cast(&d_cost[0]), thrust::raw_pointer_cast(&d_update_cost[0]), graph->edges.size())));

        //copy back mask
        mask = d_mask;
    }

    // no need to clean up vectors as they get de-allocated when they go out of scope
    previous_nodes = d_previous_node;
    cost = d_cost;
    update_cost = d_update_cost;

    std::vector<int> ret_previous_nodes(previous_nodes.size());
    thrust::copy(previous_nodes.begin(), previous_nodes.end(), ret_previous_nodes.begin());

    std::vector<int> ret_cost(cost.size());
    thrust::copy(cost.begin(), cost.end(), ret_cost.begin());

    return std::make_shared<Paths>(Paths(ret_previous_nodes, ret_cost, source_node, graph));
}