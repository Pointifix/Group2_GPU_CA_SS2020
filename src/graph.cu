#include "graph.h"

 /**
  * Generates a graph instance.
  * @param edges Edges
  * @param destinations Destinations
  * @param weights Weights
  */
Graph::Graph(std::vector<int> &edges, std::vector<int> &destinations, std::vector<int> &weights) :
    edges(std::move(edges)), destinations(std::move(destinations)), weights(std::move(weights))
{
}

std::string Graph::toString() {
    std::string builder = "Graph (" + std::to_string(edges.size()) + " vertices, " + std::to_string(destinations.size()) + " edges)";

    builder += "\nEdges:\n";
    for (const auto &edge : edges) {
        builder += std::to_string(edge) + ", ";
    }
    builder += "\nDestinations:\n";
    for (const auto &destination : destinations) {
        builder += std::to_string(destination) + ", ";
    }
    builder += "\nWeights:\n";
    for (const auto &weight : weights) {
        builder += std::to_string(weight) + ", ";
    }
    return builder + "\n";
}

Paths::Paths(std::vector<int> &previous_nodes, int source_node, std::shared_ptr<Graph> graph) :
    previous_nodes(std::move(previous_nodes)), source_node(source_node), graph(graph)
{
}

std::string Paths::toString() {
    std::string builder = "Paths (Graph with " + std::to_string(graph->edges.size()) + " vertices, " + std::to_string(graph->destinations.size()) + " edges)";

    builder += "\nSource Node:\n" + std::to_string(source_node);

    builder += "\nP:\n";
    for (const auto &previous_node : previous_nodes) {
        builder += std::to_string(previous_node) + ", ";
    }
    return builder + "\n";
}

std::vector<int> Paths::getPath(int destination)
{
    std::vector<int> path;
    path.push_back(destination);

    int current_node = destination;

    while(current_node != source_node)
    {
        current_node = previous_nodes[current_node];
        path.push_back(current_node);
    }

    std::reverse(std::begin(path), std::end(path));

    return path;
}

bool Paths::isEqualTo(const Paths* path) {
    return this->previous_nodes.size() == path->previous_nodes.size() && std::equal(this->previous_nodes.begin(),
                      this->previous_nodes.begin() + this->previous_nodes.size(), path->previous_nodes.begin());
}
