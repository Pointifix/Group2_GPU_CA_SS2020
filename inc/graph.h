#ifndef GRAPH_H
#define GRAPH_H

#include "common.cuh"

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>

class Graph {
public:
    /**
     * Encapsulates a graph represented in sparse matrix form
     * @param edges
     * @param destinations
     * @param weights
     */
    Graph(std::vector<pos_t> &edges, std::vector<pos_t> &destinations, std::vector<weight_t> &weights);

    std::vector<pos_t> edges;
    std::vector<pos_t> destinations;
    std::vector<weight_t> weights;

    std::string toString();

    std::vector<std::vector<weight_t>> printAdjacencyMatrix();
};

class Paths {
public:
    /**
     * Encapsulates all paths in a graph from a single source
     * @param previous_nodes
     * @param source_node
     * @param graph
     */
    Paths(std::vector<pos_t> &previous_nodes, std::vector<weight_t> &costs, pos_t source_node, std::shared_ptr<Graph> graph);

    std::vector<weight_t>costs;
    std::vector<pos_t>previous_nodes;
    pos_t source_node;
    std::shared_ptr<Graph> graph;

    std::string toString();

    /**
     * Compares two paths, returns d
     *
     * d = 0: Paths have equal costs and previous nodes
     * d > 0: Paths have equal costs but found d different paths
     * d = -1: Paths found different paths with different costs
     * d = -2: previous nodes array lengths differ
     *
     * @param path
     * @return
     */
    int isEqualTo(const Paths* path);

    std::vector<pos_t> getPath(pos_t destination);
};

#endif /* GRAPH_H */