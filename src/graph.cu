#include "graph.h"

 /**
  * Generates a graph instance.
  * @param edges Edges
  * @param destinations Destinations
  * @param weights Weights
  */
Graph::Graph(std::vector<uint> &edges, std::vector<uint> &destinations, std::vector<uint> &weights) :
    edges(std::move(edges)), destinations(std::move(destinations)), weights(std::move(weights))
{
}

std::string Graph::to_string() {
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

Path::Path(std::vector<uint> &edges, std::vector<uint> &destinations, std::vector<uint> &weights,
        uint source_node, uint destination_node) :
        Graph(edges, destinations, weights),
        source_node(source_node), destination_node(destination_node)
{
}
