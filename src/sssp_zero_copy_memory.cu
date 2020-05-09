#include "sssp_zero_copy_memory.h"

SSSP_Zero_Copy_Memory::SSSP_Zero_Copy_Memory(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::shared_ptr<Paths> SSSP_Zero_Copy_Memory::compute(int source_node)
{
    // TODO
    return nullptr;
}