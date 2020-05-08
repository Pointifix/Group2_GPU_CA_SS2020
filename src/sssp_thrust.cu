#include "sssp_thrust.h"

SSSP_Thrust::SSSP_Thrust(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::vector<std::vector<int>> SSSP_Thrust::compute(int source_node)
{
    // TODO
    return std::vector<std::vector<int>>();
}