#include "sssp_pinned_memory.h"

#include <utility>

SSSP_Pinned_Memory::SSSP_Pinned_Memory(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::vector<std::vector<int>> SSSP_Pinned_Memory::compute(int source_node)
{
    // TODO
    return std::vector<std::vector<int>>();
}