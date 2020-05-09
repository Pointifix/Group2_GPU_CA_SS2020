#include "sssp_pinned_memory.h"

#include <utility>

SSSP_Pinned_Memory::SSSP_Pinned_Memory(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::shared_ptr<Paths> SSSP_Pinned_Memory::compute(int source_node)
{
    // TODO
    return nullptr;
}