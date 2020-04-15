#pragma once

#include <utility>
#include <vector>
#include <string>

class Graph {
public:
    Graph(std::vector<int> &edges, std::vector<int> &destinations, std::vector<int> &weights);
    std::vector<int> edges;
    std::vector<int> destinations;
    std::vector<int> weights;

    std::string to_string()
    {
        std::string builder = "Graph (" + std::to_string(edges.size()) + " vertices, " + std::to_string(destinations.size()) + " edges)";

        builder += "\nE: ";
        for (const auto &edge : edges) {
            builder += std::to_string(edge) + ", ";
        }
        builder += "\nD: ";
        for (const auto &destination : destinations) {
            builder += std::to_string(destination) + ", ";
        }
        builder += "\nW: ";
        for (const auto &weight : weights) {
            builder += std::to_string(weight) + ", ";
        }
        return builder + "\n";
    }
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
