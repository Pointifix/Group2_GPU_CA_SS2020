#include "graph.h"

#include <utility>

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

Path::Path(std::vector<int> &edges, std::vector<int> &destinations, std::vector<int> &weights,
        int source_node, int destination_node) :
        Graph(edges, destinations, weights),
        source_node(source_node), destination_node(destination_node)
{
}
