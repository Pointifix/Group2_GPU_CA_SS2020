#ifndef GRAPH_H
#define GRAPH_H

#include <utility>
#include <vector>
#include <string>

class Graph {
public:
    Graph(std::vector<uint> &edges, std::vector<uint> &destinations, std::vector<uint> &weights);
    std::vector<uint> edges;
    std::vector<uint> destinations;
    std::vector<uint> weights;

    std::string toString();
};

class Path : public Graph {
public:
    /**
     * Creates a Path instance.
     * @param edges Edges
     * @param destinations Destinations
     * @param weights Weights
     * @param source_node Source Node (index)
     * @param destination_node Destination Node (index)
     */
    Path(std::vector<uint> &edges, std::vector<uint> &destinations, std::vector<uint> &weights,
            uint source_node, uint destination_node);
    uint source_node;
    uint destination_node;
};

#endif /* GRAPH_H */