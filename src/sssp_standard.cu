#include "sssp_standard.h"

SSSP_Standard::SSSP_Standard(std::shared_ptr<Graph> graph, MemoryType memType, SearchType searchType) :
    SSSP(std::move(graph)), m_memType(memType), m_searchType(searchType) {
}

std::shared_ptr<Paths> SSSP_Standard::compute(int source_node)
{
    size_t numNodes = graph->edges.size();
    size_t numEdges = graph->destinations.size();

    size_t sizeNodes = numNodes * sizeof(data_t);
    size_t sizeEdges = numEdges * sizeof(data_t);

    data_t *d_edges = nullptr;
    data_t *d_destinations = nullptr;
    data_t *d_weights = nullptr;

    mask_t *d_mask = nullptr;
    size_t sizeMask = numNodes * sizeof(mask_t);

    data_t *d_previous_node = nullptr;
    data_t *d_cost = nullptr;

    mask_t *mask = nullptr;
    std::function<bool()> maskContainsTrue;
    switch(m_searchType) {
        case GPU:
            // TODO
            M_A(false);
            std::terminate();
            break;
        case CPU:
        default:
            switch (m_memType) {
                case ZERO_COPY:
                    // TODO
                    M_A(false);
                    std::terminate();
                    break;
                case PINNED:
                    M_C(cudaMallocHost((void**) &mask, sizeMask));
                    break;
                case NORMAL:
                default:
                    mask = (bool*) malloc(sizeMask);
                    break;
            }
            const mask_t *maskFirst = &mask[0];
            const mask_t *maskLast = &mask[numNodes];
            maskContainsTrue = [maskFirst, maskLast]() {
                return std::find(maskFirst, maskLast, true) != maskLast;
            };
            break;
    }

    // If mask is NOT allocated and we're in debug mode, allocate it!
    M_D(if (mask == nullptr) M_C(cudaMallocHost((void**) &mask, sizeMask)));

    M_C(cudaMalloc((void**) &d_edges,         sizeNodes));
    M_C(cudaMalloc((void**) &d_destinations,  sizeEdges));
    M_C(cudaMalloc((void**) &d_weights,       sizeEdges));
    M_C(cudaMalloc((void**) &d_mask,          sizeMask));
    M_C(cudaMalloc((void**) &d_previous_node, sizeNodes));
    M_C(cudaMalloc((void**) &d_cost,          sizeNodes));

    M_C(cudaMemcpy(d_edges,        graph->edges.data(),        sizeNodes, cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_destinations, graph->destinations.data(), sizeEdges, cudaMemcpyHostToDevice));
    M_C(cudaMemcpy(d_weights,      graph->weights.data(),      sizeEdges, cudaMemcpyHostToDevice));
    alg::fill_parcu(d_mask, numNodes, false);
    alg::fill_parcu(d_previous_node, numNodes, M_INVALID);
    alg::fill_parcu(d_cost, numNodes, std::numeric_limits<data_t>::max());

    alg::set_parcu(d_mask, source_node, true);
    alg::set_parcu(d_cost, source_node, 0);

    // Debug: copy initialized mask to host to check the values
    M_DC(cudaMemcpy(mask, d_mask, sizeMask, cudaMemcpyDeviceToHost));

    // while we still find true in the mask (Ma not empty)
    while (maskContainsTrue())
    {
        int numBlocks = ceil((double)graph->edges.size() / M_BLOCKSIZE);

        M_CFUN((alg::SSSP_Kernel<<<numBlocks, M_BLOCKSIZE>>>(d_edges, d_destinations, d_weights,
                d_previous_node, d_mask, d_cost, graph->edges.size(), graph->destinations.size())));

        if (m_searchType == CPU || M_DEBUG) {
            //copy back mask
            M_C(cudaMemcpy(mask, d_mask, sizeMask, cudaMemcpyDeviceToHost));
        }
    }

    std::vector<data_t> previous_nodes(numNodes);
    std::vector<data_t> cost(numNodes);
    M_C(cudaMemcpy(previous_nodes.data(), d_previous_node, sizeNodes, cudaMemcpyDeviceToHost));
    M_C(cudaMemcpy(cost.data(), d_cost, sizeNodes, cudaMemcpyDeviceToHost));

    M_C(cudaFree(d_edges));
    M_C(cudaFree(d_destinations));
    M_C(cudaFree(d_weights));
    M_C(cudaFree(d_previous_node));
    M_C(cudaFree(d_mask));
    M_C(cudaFree(d_cost));

    std::shared_ptr<Paths> paths = std::make_shared<Paths>(Paths( previous_nodes, cost, source_node, graph));

    return paths;
}