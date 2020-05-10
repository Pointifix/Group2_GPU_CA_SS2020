#ifndef GRAPH_H
#define GRAPH_H

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

class Graph {
public:
    /**
     * Encapsulates a graph represented in sparse matrix form
     * @param edges
     * @param destinations
     * @param weights
     */
    Graph(std::vector<int> &edges, std::vector<int> &destinations, std::vector<int> &weights);

    std::vector<int> edges;
    std::vector<int> destinations;
    std::vector<int> weights;

    std::string toString();
};

class Paths {
public:
    /**
     * Encapsulates all paths in a graph from a single source
     * @param previous_nodes
     * @param source_node
     * @param graph
     */
    Paths(std::vector<int> &previous_nodes, std::vector<int> &costs, int source_node, std::shared_ptr<Graph> graph);

    std::vector<int>previous_nodes;
    std::vector<int> costs;
    int source_node;
    std::shared_ptr<Graph> graph;

    std::string toString();

    bool isEqualTo(const Paths* path);

    std::vector<int> getPath(int destination);
};

#endif /* GRAPH_H */