#include "sssp_zero_copy_memory.h"

SSSP_Zero_Copy_Memory::SSSP_Zero_Copy_Memory(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::vector<std::shared_ptr<Path>> SSSP::compute(int source_node)
{
    // TODO
}