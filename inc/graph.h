#ifndef GRAPH_H
#define GRAPH_H

#include <utility>
#include <vector>
#include <string>

class Graph {
public:
    Graph(std::vector<int> &edges, std::vector<int> &destinations, std::vector<int> &weights);
    std::vector<int> edges;
    std::vector<int> destinations;
    std::vector<int> weights;

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
    Path(std::vector<int> &edges, std::vector<int> &destinations, std::vector<int> &weights,
            int source_node, int destination_node);
    int source_node;
    int destination_node;
};

#endif /* GRAPH_H */