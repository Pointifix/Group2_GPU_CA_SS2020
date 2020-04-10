#pragma once

#include <memory>
#include "graph.h"

/**
 * Generates a connected Graph with given number of nodes and density.
 * @param num_nodes Number of nodes.
 * @param density Density
 * @param directed is always true (and this doesn't matter its ignored)
 * @return connected Graph (actually nullptr)
 */
std::shared_ptr<Graph> generateConnectedGraph(int num_nodes, float density, bool directed = true)
{
    // TODO
    return nullptr;
}