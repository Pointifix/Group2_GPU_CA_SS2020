#include "sssp_thrust.h"

SSSP_Thrust::SSSP_Thrust(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::shared_ptr<Paths> SSSP_Thrust::compute(int source_node)
{
    using mask_t = bool;

    thrust::host_vector<data_t> previous_nodes(graph->edges.size(), -1);
    thrust::host_vector<mask_t> mask(graph->edges.size(), 0);
    thrust::host_vector<data_t> cost(graph->edges.size(), std::numeric_limits<int>::max());

    mask[source_node] = true;
    cost[source_node] = 0;

    thrust::device_vector<data_t> d_edges = graph->edges;
    thrust::device_vector<data_t> d_destinations = graph->destinations;
    thrust::device_vector<data_t> d_weights = graph->weights;
    thrust::device_vector<data_t> d_previous_node = previous_nodes;
    thrust::device_vector<mask_t> d_mask = mask;
    thrust::device_vector<data_t> d_cost = cost;

    // while we still find false in the mask (Ma not empty)
    while (std::find(mask.begin(), mask.end(), true) != mask.end())
    {
        int numBlocks = ceil((double)graph->edges.size() / 256);

        dim3 threadsPerBlock(256);
        M_CFUN((alg::SSSP_Kernel<<<numBlocks, threadsPerBlock>>>(
                thrust::raw_pointer_cast(&d_edges[0]), thrust::raw_pointer_cast(&d_destinations[0]),
                thrust::raw_pointer_cast(&d_weights[0]), thrust::raw_pointer_cast(&d_previous_node[0]),
                thrust::raw_pointer_cast(&d_mask[0]), thrust::raw_pointer_cast(&d_cost[0]), graph->edges.size(), graph->destinations.size())));

        //copy back mask
        mask = d_mask;
    }

    // no need to clean up vectors as they get de-allocated when they go out of scope
    previous_nodes = d_previous_node;
    cost = d_cost;

    std::vector<int> ret_previous_nodes(previous_nodes.size());
    thrust::copy(previous_nodes.begin(), previous_nodes.end(), ret_previous_nodes.begin());

    std::vector<int> ret_cost(cost.size());
    thrust::copy(cost.begin(), cost.end(), ret_cost.begin());

    return std::make_shared<Paths>(Paths(ret_previous_nodes, ret_cost, source_node, graph));
}