#include "graph.h"

 /**
  * Generates a graph instance.
  * @param edges Edges
  * @param destinations Destinations
  * @param weights Weights
  */
Graph::Graph(std::vector<pos_t> &edges, std::vector<pos_t> &destinations, std::vector<weight_t> &weights) :
    edges(std::move(edges)), destinations(std::move(destinations)), weights(std::move(weights))
{
}

std::string Graph::toString() {
    std::string builder = "Graph (" + std::to_string(edges.size()) + " vertices, " + std::to_string(destinations.size()) + " edges)";

    builder += "\nEdges:\n\t";
    for (const auto &edge : edges) {
        builder += std::to_string(edge) + ",\t";
    }
    builder += "\nDestinations:\n\t";
    for (const auto &destination : destinations) {
        builder += std::to_string(destination) + ",\t";
    }
    builder += "\nWeights:\n\t";
    for (const auto &weight : weights) {
        builder += std::to_string(weight) + ",\t";
    }
    return builder + "\n";
}

std::vector<std::vector<weight_t>> Graph::printAdjacencyMatrix()
{
    std::vector<std::vector<weight_t>> adjacencyMatrix(edges.size(), std::vector<weight_t>(edges.size(), 0));

    for(pos_t i = 0; i < edges.size(); i++)
    {
        pos_t first = edges[i];
        pos_t last = (i + 1 < edges.size()) ? edges[i + 1] : destinations.size();

        for(pos_t j = first; j < last; j++)
        {
            adjacencyMatrix[i][destinations[j]] = weights[j];
        }
    }

    for(pos_t i = 0; i < edges.size(); i++)
    {
        for(pos_t j = 0; j < edges.size(); j++)
        {
            std::cout << adjacencyMatrix[i][j] << ", ";
        }
        std::cout << std::endl;
    }

    return adjacencyMatrix;
}

Paths::Paths(std::vector<pos_t> &previous_nodes, std::vector<weight_t> &costs, pos_t source_node, std::shared_ptr<Graph> graph) :
    previous_nodes(std::move(previous_nodes)), costs(costs), source_node(source_node), graph(graph)
{
}

std::string Paths::toString() {
    std::string builder = "Paths (Graph with " + std::to_string(graph->edges.size()) + " vertices, " + std::to_string(graph->destinations.size()) + " edges)";

    builder += "\nSource Node:\n\t" + std::to_string(source_node);

    builder += "\nPrevious Nodes:\n\t";
    for (const auto &previous_node : previous_nodes) {
        builder += std::to_string(previous_node) + ",\t";
    }
    builder += "\nCosts:\n\t";
    for (const auto &cost : costs) {
        if (cost == std::numeric_limits<int>::max()) builder += "inf,\t";
        else builder += std::to_string(cost) + ",\t";
    }
    return builder + "\n";
}

std::vector<pos_t> Paths::getPath(pos_t destination)
{
    std::vector<pos_t> path;
    path.push_back(destination);

    pos_t current_node = destination;

    while(previous_nodes[current_node] != -1 && current_node != source_node)
    {
        current_node = previous_nodes[current_node];
        path.push_back(current_node);
    }

    std::reverse(std::begin(path), std::end(path));

    return path;
}

int Paths::isEqualTo(const Paths* path) {
    if (this->previous_nodes.size() != path->previous_nodes.size()) return -2;

    std::vector<int> difference;

    for (size_t i = 0; i < this->previous_nodes.size(); i++)
    {
        if (this->previous_nodes.at(i) != path->previous_nodes.at(i)) difference.push_back(i);
    }

    for (pos_t i = 0; i < difference.size(); i++)
    {
        if (this->costs.at(difference.at(i)) != path->costs.at(difference.at(i))) return -1;
    }

    return difference.size();
}
