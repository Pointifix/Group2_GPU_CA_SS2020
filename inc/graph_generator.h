#pragma once

#include <memory>

#include "graph.h"

/**
 * Generates a connected Graph with given number of nodes and density.
 * @param num_nodes Number of nodes
 * @param density Density in the range [0,1]
 * @param directed is always true (and this doesn't matter its ignored)
 * @return connected Graph (actually nullptr)
 */
std::shared_ptr<Graph> generateConnectedGraph(int num_nodes, float density, bool directed = true)
{
    if (num_nodes < 0 || density < 0 || density > 1) {
        return nullptr;
    }

    // https://en.wikipedia.org/wiki/Dense_graph
    // The graph density of simple graphs is defined to be the ratio of the number of edges |E| with respect to the
    //   maximum possible edges.
    // For directed simple graphs, the maximum possible edges is twice that of undirected graphs to account for the
    //   directedness
    int num_edges = (int) (density * (float)num_nodes * (float)(num_nodes - 1) * (directed ? 1 : 0.5));

    std::vector<int> edges(num_nodes);
    std::vector<int> directions(num_edges);
    std::vector<int> weights(num_edges);

    // TODO

    return std::make_shared<Graph>(Graph(edges, directions, weights));;
}