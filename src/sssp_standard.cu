#include "sssp_standard.h"

SSSP_Standard::SSSP_Standard(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::vector<std::shared_ptr<Path>> SSSP_Standard::compute(int source_node)
{
    // TODO
    return std::vector<std::shared_ptr<Path>>();
}