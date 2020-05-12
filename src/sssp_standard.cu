#include "sssp_standard.h"

SSSP_Standard::SSSP_Standard(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}



std::shared_ptr<Paths> SSSP_Standard::compute(int source_node)
{
    std::vector<int> previous_nodes(graph->edges.size(), -1);
    std::vector<int> mask(graph->edges.size(), 0);
    std::vector<int> cost(graph->edges.size(), std::numeric_limits<int>::max());

    mask.at(source_node) = true;
    cost.at(source_node) = 0;

    int *d_edges = nullptr;
    int *d_destinations = nullptr;
    int *d_weights = nullptr;
    int *d_previous_node = nullptr;
    int *d_mask = nullptr;
    int *d_cost = nullptr;

    M_C(cudaMalloc((void**) &d_edges,          graph->edges.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_destinations,   graph->destinations.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_weights,        graph->weights.size() * sizeof(int)));

    M_C(cudaMalloc((void**) &d_previous_node, previous_nodes.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_mask, mask.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_cost, cost.size() * sizeof(int)));

    M_C(cudaMemcpy(d_edges,        &graph->edges[0],        graph->edges.size() * sizeof(int),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_destinations, &graph->destinations[0], graph->destinations.size() * sizeof(int),   cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_weights,      &graph->weights[0],      graph->weights.size() * sizeof(int),        cudaMemcpyHostToDevice));

    M_C(cudaMemcpy(d_previous_node,&previous_nodes[0],  previous_nodes.size() * sizeof(int),cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_mask,         &mask[0],            mask.size() * sizeof(int),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_cost,         &cost[0],            cost.size() * sizeof(int),          cudaMemcpyHostToDevice));

    // while we still find false in the mask (Ma not empty)
    while (std::find(mask.begin(), mask.end(), true) != mask.end())
    {
        int numBlocks = ceil((double)graph->edges.size() / 256);

        dim3 threadsPerBlock(256);
        M_CFUN((alg::SSSP_Kernel<<<numBlocks, threadsPerBlock>>>(d_edges, d_destinations, d_weights,
                d_previous_node, d_mask, d_cost, graph->edges.size(), graph->destinations.size())));

        //copy back mask
        M_C(cudaMemcpy(&mask[0], d_mask, mask.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }

    M_C(cudaMemcpy(&previous_nodes[0], d_previous_node, previous_nodes.size() * sizeof(int), cudaMemcpyDeviceToHost));
    M_C(cudaMemcpy(&cost[0], d_cost, cost.size() * sizeof(int), cudaMemcpyDeviceToHost));

    M_C(cudaFree(d_edges));
    M_C(cudaFree(d_destinations));
    M_C(cudaFree(d_weights));
    M_C(cudaFree(d_previous_node));
    M_C(cudaFree(d_mask));
    M_C(cudaFree(d_cost));

    std::shared_ptr<Paths> paths = std::make_shared<Paths>(Paths(previous_nodes, cost, source_node, graph));

    return paths;
}