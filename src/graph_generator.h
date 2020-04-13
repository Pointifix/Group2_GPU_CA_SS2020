#pragma once

#include <memory>
#include <stdlib.h>
#include <time.h>

#include "graph.h"
/**
 * Generates a connected Graph with given number of nodes and density.
 * @param num_nodes Number of nodes
 * @param density Density in the range [0,1], minimum Density for the graph to be connect Emin = 1 / |V|
 * @param directed is always true (and this doesn't matter its ignored)
 * @param max_weight maximum weight value possible for edge weights
 * @return connected Graph (actually nullptr)
 */
std::shared_ptr<Graph> generateConnectedGraph(int num_nodes, float density, bool directed = true, int max_weight = 10)
{
    if (num_nodes < 0 || density < 0 || density > 1) {
        return nullptr;
    }

    // https://en.wikipedia.org/wiki/Dense_graph
    // The graph density of simple graphs is defined to be the ratio of the number of edges |E| with respect to the
    //   maximum possible edges.
    // For directed simple graphs, the maximum possible edges is twice that of undirected graphs to account for the
    //   directedness
    int num_edges = (int) (density * (float)num_nodes * (float)(num_nodes - 1) * (directed ? 1 : 0.5));

    std::vector<int> edges(num_nodes);
    std::vector<int> directions(num_edges);
    std::vector<int> weights(num_edges);

	srand(time(NULL));
	std::vector<int> connected_nodes;
	std::vector<int> not_connected_nodes(num_nodes);
	std::vector<std::vector<int>>directions_builder(num_nodes);

	for (size_t i = 0; i < num_nodes; i++) not_connected_nodes.at(i) = i;

	int start_node = rand() % num_nodes;
	connected_nodes.push_back(start_node);
	not_connected_nodes.erase(not_connected_nodes.begin() + start_node);

	for (size_t i = 0; i < num_edges; i++)
	{
		int random_source = connected_nodes.at(rand() % connected_nodes.size());
		int random_destination = rand() % num_nodes;

		if (!not_connected_nodes.empty())
		{
			int random_not_connected_node = rand() % not_connected_nodes.size();
			random_destination = not_connected_nodes.at(random_not_connected_node);
			not_connected_nodes.erase(not_connected_nodes.begin() + random_not_connected_node);
			connected_nodes.push_back(random_destination);
		}
		else
		{
			while (random_source == random_destination)	random_destination = rand() % num_nodes;
		}

		if (rand() % 2)
		{
			int temp = random_destination;
			random_destination = random_source;
			random_source = temp;
		}

		if (std::find(directions_builder.at(random_source).begin(), directions_builder.at(random_source).end(), random_destination) != directions_builder.at(random_source).end()) i--;
		else directions_builder.at(random_source).push_back(random_destination);
	}

	// assign random weights
	for (size_t i = 0; i < weights.size(); i++)
	{
		weights.at(i) = rand() % (max_weight + 1);
	}

	// copy random directions into directions vector
	size_t i = 0;
	size_t j = 0;
	size_t k = 0;
	for (auto && node_directions : directions_builder) {
		edges.at(k) = i;
		for (auto && direction : node_directions)
		{
			directions.at(i + j) = node_directions.at(j);
			j++;
		}
		j = 0;
		i += node_directions.size();
		k++;
	}

    return std::make_shared<Graph>(Graph(edges, directions, weights));;
}