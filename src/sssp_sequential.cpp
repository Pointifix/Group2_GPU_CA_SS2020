#include "sssp_sequential.h"

SSSP_Sequential::SSSP_Sequential(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::vector<std::shared_ptr<Path>> SSSP_Sequential::compute(int source_node)
{
    // TODO
    return std::vector<std::shared_ptr<Path>>();
}