#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include <memory>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <list>

#include "graph.h"
namespace graphgen {
    /**
     * Generates a connected Graph with given number of nodes and density, if the density is too small for the graph to be weakly connected it return a disconnected graph.
     * @param num_nodes Number of nodes
     * @param density Density in the range [0,1], minimum Density for the graph to be connected, Emin = 1 / |V|, equivalent to p of a Gilbert graph G(n,p)
     * @param directed is always true (and this doesn't matter its ignored) TODO should it be?
     * @param max_weight maximum weight value possible for edge weights
     * @return connected Graph
     */
    std::shared_ptr<Graph> generateConnectedGraph(int num_nodes, float density,
                                                  bool directed = true, int max_weight = 10,
                                                  unsigned int seed = time(nullptr)) {
        if (num_nodes < 2 || density < 0 || density > 1) {
            return nullptr;
        }

        // https://en.wikipedia.org/wiki/Dense_graph
        // The graph density of simple graphs is defined to be the ratio of the number of edges |E| with respect to the
        //   maximum possible edges.
        // For directed simple graphs, the maximum possible edges is twice that of undirected graphs to account for the
        //   directedness
        const int num_edges = (int) (density * (float) num_nodes * (float) (num_nodes - 1) * (directed ? 1 : 0.5));

        std::vector<int> edges(num_nodes);
        std::vector<int> directions(num_edges);
        std::vector<int> weights(num_edges);

        // init random number generator, create building vectors
        srand(seed);
        std::vector<int> connected_nodes;
        std::vector<int> not_connected_nodes(num_nodes);
        std::vector<std::vector<int>> directions_builder(num_nodes);

        // init vector with all nodes
        for (int i = 0; i < num_nodes; i++) {
            not_connected_nodes.at(i) = i;
        }

        // pick a random node which is the start of the connected nodes set
        int start_node = rand() % num_nodes;
        connected_nodes.push_back(start_node);
        not_connected_nodes.erase(not_connected_nodes.begin() + start_node);

        // first, pick random non-connected nodes and connect them with connected nodes until all nodes are connected (weakly connected graph)
        // when every node has at least one in or outgoing edge pick random edges until the density is satisfied
        for (int i = 0; i < num_edges; i++) {
            int random_source = connected_nodes.at(rand() % connected_nodes.size());
            int random_destination = rand() % num_nodes;

            if (!not_connected_nodes.empty()) {
                int random_not_connected_node = rand() % not_connected_nodes.size();
                random_destination = not_connected_nodes.at(random_not_connected_node);
                not_connected_nodes.erase(not_connected_nodes.begin() + random_not_connected_node);
                connected_nodes.push_back(random_destination);
            } else {
                while (random_source == random_destination || directions_builder.at(random_source).size() == num_nodes ||
                       std::find(directions_builder.at(random_source).begin(), directions_builder.at(random_source).end(),
                                 random_destination) != directions_builder.at(random_source).end())
                {
                    random_source = connected_nodes.at(rand() % connected_nodes.size());
                    random_destination = rand() % num_nodes;
                }
            }

            directions_builder.at(random_source).push_back(random_destination);
        }

        // assign random weights
        for (int & weight : weights) {
            weight = rand() % (max_weight) + 1;
        }

        // copy random directions (vector of vectors) into directions vector and set the pointers of edge vector to the corresponding offsets
        int i = 0;
        int j = 0;
        int k = 0;
        for (auto &&node_directions : directions_builder) {
            edges.at(k) = i;
            std::sort(node_directions.begin(), node_directions.end());

            for (auto &&direction : node_directions) {
                directions.at(i + j) = node_directions.at(j);
                j++;
            }

            j = 0;
            i += node_directions.size();
            k++;
        }

        return std::make_shared<Graph>(Graph(edges, directions, weights));
    }

    /**
     * Generates a Graph with given number of nodes and density
     * @param num_nodes Number of nodes
     * @param density Density in the range [0,1], equivalent to p of a Gilbert graph G(n,p)
     * @param directed is always true (and this doesn't matter its ignored) TODO should it be?
     * @param max_weight maximum weight value possible for edge weights
     * @return connected Graph
     */
    std::shared_ptr<Graph> generateGraph(int num_nodes, float density,
                                                   bool directed = true, int max_weight = 10,
                                                   unsigned int seed = time(nullptr)) {
        if (num_nodes < 2 || density < 0 || density > 1) {
            return nullptr;
        }

        // https://en.wikipedia.org/wiki/Dense_graph
        // The graph density of simple graphs is defined to be the ratio of the number of edges |E| with respect to the
        //   maximum possible edges.
        // For directed simple graphs, the maximum possible edges is twice that of undirected graphs to account for the
        //   directedness
        const int num_edges = (int) (density * (float) num_nodes * (float) (num_nodes - 1) * (directed ? 1 : 0.5));

        std::vector<int> edges(num_nodes);
        std::vector<int> destinations(num_edges, -1);
        std::vector<int> weights(num_edges);

        int edges_remaining = num_edges;
        std::vector<int> edges_count(num_nodes, 0);

        srand(seed);

        // distribute how many edges each node has randomly
        while(edges_remaining > 0)
        {
            int random_node = rand() % num_nodes;

            if (edges_count[random_node] < num_nodes - 1)
            {
                edges_count[random_node]++;
                edges_remaining--;
            }
        }

        // assign edge references
        int current_destination = 0;
        for(int i = 0; i < num_nodes; i++)
        {
            edges.at(i) = current_destination;
            current_destination += edges_count.at(i);
        }

        // assign random destinations
        int random_node;

        for (int i = 0; i < num_nodes; i++)
        {
            int first = edges[i];
            int last = (i + 1 < edges.size()) ? edges[i + 1] : destinations.size();

            for (int j = first; j < last; j++)
            {
                do random_node = rand() % num_nodes; while (random_node == i || std::find(destinations.begin() + first, destinations.begin() + last, random_node) != destinations.begin() + last);

                destinations.at(j) = random_node;
            }

            std::sort(destinations.begin() + first, destinations.begin() + last);
        }

        // assign random weights
        for (int i = 0; i < num_edges; i++) weights.at(i) = rand() % max_weight;

        return std::make_shared<Graph>(Graph(edges, destinations, weights));
    }
}

#endif /* GRAPH_GENERATOR_H */