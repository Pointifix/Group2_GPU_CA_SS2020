#include <iostream>
#include "sssp_sequential.h"

SSSP_Sequential::SSSP_Sequential(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::vector<std::shared_ptr<Path>> SSSP_Sequential::compute(int source_node)
{
    int N = graph->edges.size();
    std::vector<int> cluster;
    std::vector<int> cost(N, std::numeric_limits<int>::max());
    cost[source_node] = 0;
    std::vector<int> used_edge(N, -1);

    // we initially add the source node and update the costs
    cluster.push_back(source_node);

    std::cout << "picked source: " << source_node << std::endl;

    // either we take all edges until the end or until the next node stores its edges
    size_t edge_list_end = (source_node + 1 < graph->edges.size()) ? graph->edges[source_node + 1] : graph->destinations.size();

    // for all edges incident from this node
    for(size_t i = graph->edges[source_node]; i < graph->destinations.size() && i < edge_list_end; i++)
    {
        // node that the edge is incident to (s) -> (dest)
        size_t dest_node = graph->destinations[i];
        cost[dest_node] = graph->weights[i]; // the cost is the weight of the edge
        used_edge[dest_node] = i; // we remember the index of the edge we use
    }

    do
    {
        // pick the node with the smallest cost
        int min = std::numeric_limits<int>::max();
        int picked_node = -1;
        for(int i = 0; i < cost.size(); i++)
        {
            // if the cost is the smallest and we didn't pick this node yet
            if(cost[i] < min && std::find(cluster.begin(), cluster.end(), i) == cluster.end())
            {
                picked_node = i;
                min = cost[i];
            }
        }

        // we cannot pick node -1 lets stop or burn and crash
        if(picked_node == -1)
        {
            break;
        }

        std::cout << "picked node: " << picked_node << std::endl;

        // we add this node to the cluster (index of the picked node)
        cluster.push_back(picked_node);
        // we select the edge-list (index of the edge list)
        size_t edge_list_index_start = graph->edges[picked_node];

        // either we take all edges until the end or until the next node stores its edges
        // source_node+1 = following node if it exists
        edge_list_end = (picked_node + 1 < graph->edges.size()) ? graph->edges[picked_node + 1] : graph->destinations.size();

        // we update the costs for all edges from this node
        // until we reach the end of the destinations array or the edges of the next node start
        for(size_t i = edge_list_index_start; i < graph->destinations.size() && i < edge_list_end; i++)
        {
            // the node targeted by this edge
            size_t dest_node = graph->destinations[i];

            // if we already picked it we don't need to do anything, continue with the next one
            if(std::find(cluster.begin(), cluster.end(), dest_node) != cluster.end()) {
                continue;
            }

            // cost to the picked node + weight of the current edge < cost for the destined node ?
            if(cost[picked_node] + graph->weights[i] < cost[dest_node])
            {
                cost[dest_node] = cost[picked_node] + graph->weights[i]; // lets update its cost
                used_edge[dest_node] = i;
            }
        }
    } while(cluster.size() < graph->edges.size()); // while there is a vertex that is not reached yet

    std::cout << "Single Shortest Path" << std::endl;
    for(int i = 0; i < cluster.size(); i++)
    {
        std::cout << cluster[i] << " with edge " << used_edge[cluster[i]] << ", ";
    }

    std::vector<std::shared_ptr<Path>> shortest_paths;

    std::vector<int> e(graph->edges.size(), 0);
    std::vector<int> d;
    std::vector<int> w;

    int runner = 0;
    for(int i = 1; i < cluster.size(); i++) // for each new node we can create a new path (source_node to destination_node)
    {
        std::shared_ptr<Path> p;
        p->source_node = source_node;
        p->destination_node = cluster[i];
    }

    return std::vector<std::shared_ptr<Path>>();
}