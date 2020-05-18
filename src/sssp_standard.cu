#include "sssp_standard.h"

SSSP_Standard::SSSP_Standard(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::shared_ptr<Paths> SSSP_Standard::compute(int source_node) {
    size_t numNodes = graph->edges.size();
    size_t numEdges = graph->destinations.size();

    // Sizes
    size_t sizeNodes = numNodes * sizeof(pos_t);
    size_t sizeEdges = numEdges * sizeof(pos_t);
    size_t sizeWeights = numEdges * sizeof(weight_t);
    size_t sizeMask = numNodes * sizeof(mask_t);
    size_t sizeCost = numNodes * sizeof(weight_t);

    // Device memory
    pos_t *d_edges = nullptr;
    pos_t *d_destinations = nullptr;
    weight_t *d_weights = nullptr;
    mask_t *d_mask = nullptr;
    pos_t *d_previous_node = nullptr;
    weight_t *d_cost = nullptr;

    // Allocate host memory
    auto *mask = new mask_t[numNodes];
    auto *previous_nodes = new pos_t[numNodes];
    auto *cost = new weight_t[numNodes];

    // Allocate d_previous_node and d_cost no matter the mode
    M_C(cudaMalloc((void **) &d_previous_node, sizeNodes));
    M_C(cudaMalloc((void **) &d_cost, sizeCost));
    M_C(cudaMalloc((void **) &d_mask, sizeMask));

    M_C(cudaMalloc((void **) &d_edges, sizeNodes));
    M_C(cudaMalloc((void **) &d_destinations, sizeEdges));
    M_C(cudaMalloc((void **) &d_weights, sizeWeights));
    M_C(cudaMemcpy(d_edges, graph->edges.data(), sizeNodes, cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_destinations, graph->destinations.data(), sizeEdges, cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_weights, graph->weights.data(), sizeWeights, cudaMemcpyHostToDevice));

    alg::fill_parcu(d_mask, numNodes, M_MASK_FALSE);
    alg::fill_parcu(d_previous_node, numNodes, M_INVALID_POSITION);
    alg::fill_parcu(d_cost, numNodes, std::numeric_limits<weight_t>::max());

    alg::set_parcu(d_mask, source_node, M_MASK_TRUE);
    alg::set_parcu(d_cost, source_node, 0);

    // while we still find true in the mask (Ma not empty)
    const mask_t *maskFirst = &mask[0];
    const mask_t *maskLast = &mask[numNodes];
    do {
        int numBlocks = ceil((double) graph->edges.size() / M_BLOCKSIZE);

        M_CFUN((alg::SSSP_Kernel<<<numBlocks, M_BLOCKSIZE>>>(d_edges, d_destinations, d_weights,
                       d_previous_node, d_mask, d_cost, graph->edges.size(), graph->destinations.size())));

        //copy back mask
        M_C(cudaMemcpy(mask, d_mask, sizeMask, cudaMemcpyDeviceToHost));
    } while (std::find(maskFirst, maskLast, true) != maskLast);

    M_C(cudaMemcpy(previous_nodes, d_previous_node, sizeNodes, cudaMemcpyDeviceToHost));
    M_C(cudaMemcpy(cost, d_cost, sizeCost, cudaMemcpyDeviceToHost));
    std::vector<pos_t> ret_previous_nodes(previous_nodes, previous_nodes + graph->edges.size());
    std::vector<weight_t> ret_cost(cost, cost + graph->edges.size());

    M_C(cudaFree(d_edges));
    M_C(cudaFree(d_destinations));
    M_C(cudaFree(d_weights));
    M_C(cudaFree(d_previous_node));
    M_C(cudaFree(d_cost));
    M_C(cudaFree(d_mask));

    delete[] mask;
    delete[] previous_nodes;
    delete[] cost;

    std::shared_ptr<Paths> paths = std::make_shared<Paths>(Paths(ret_previous_nodes, ret_cost, source_node, graph));

    return paths;
}