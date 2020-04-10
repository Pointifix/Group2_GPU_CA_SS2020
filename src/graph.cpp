#include "graph.h"

 /**
  * Generates a graph instance.
  * @param edges Edges
  * @param destinations Destinations
  * @param weights Weights
  */
Graph::Graph(std::vector<int> edges, std::vector<int> destinations, std::vector<int> weights)
        : edges(std::move(edges)), destinations(std::move(destinations)), weights(std::move(weights))
{
}
