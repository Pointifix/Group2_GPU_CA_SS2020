#include "sssp_standard.h"

SSSP_Standard::SSSP_Standard(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}



std::shared_ptr<Paths> SSSP_Standard::compute(int source_node)
{
    std::vector<m_t> previous_nodes(graph->edges.size(), -1);
    std::vector<int> mask(graph->edges.size(), 0);
    std::vector<m_t> cost(graph->edges.size(), std::numeric_limits<m_t>::max());

    mask.at(source_node) = true;
    cost.at(source_node) = 0;

    m_t *d_edges = nullptr;
    m_t *d_destinations = nullptr;
    m_t *d_weights = nullptr;
    m_t *d_previous_node = nullptr;
    int *d_mask = nullptr;
    m_t *d_cost = nullptr;

    M_C(cudaMalloc((void**) &d_edges,          graph->edges.size() * sizeof(m_t)));
    M_C(cudaMalloc((void**) &d_destinations,   graph->destinations.size() * sizeof(m_t)));
    M_C(cudaMalloc((void**) &d_weights,        graph->weights.size() * sizeof(m_t)));

    M_C(cudaMalloc((void**) &d_previous_node, previous_nodes.size() * sizeof(m_t)));
    M_C(cudaMalloc((void**) &d_mask, mask.size() * sizeof(int)));
    M_C(cudaMalloc((void**) &d_cost, cost.size() * sizeof(m_t)));

    M_C(cudaMemcpy(d_edges,        &graph->edges[0],        graph->edges.size() * sizeof(m_t),          cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_destinations, &graph->destinations[0], graph->destinations.size() * sizeof(m_t),   cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_weights,      &graph->weights[0],      graph->weights.size() * sizeof(m_t),        cudaMemcpyHostToDevice));

    M_C(cudaMemcpy(d_previous_node, previous_nodes.data(),  previous_nodes.size() * sizeof(m_t),cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_mask,          mask.data(),            mask.size() * sizeof(int),         cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_cost,          cost.data(),            cost.size() * sizeof(m_t),          cudaMemcpyHostToDevice));

    // while we still find false in the mask (Ma not empty)
    while (std::find(mask.begin(), mask.end(), true) != mask.end())
    {
        int numBlocks = ceil((double)graph->edges.size() / M_BLOCKSIZE);

        M_CFUN((alg::SSSP_Kernel<<<numBlocks, M_BLOCKSIZE>>>(d_edges, d_destinations, d_weights,
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