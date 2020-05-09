#include "sssp_sequential.h"

SSSP_Sequential::SSSP_Sequential(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::vector<std::vector<int>> SSSP_Sequential::compute(int source_node)
{
    int N = graph->edges.size();
    std::vector<int> cluster;
    std::vector<int> cost(N, std::numeric_limits<int>::max());
    std::vector<int> used_edge(N, -1);
    std::vector<int> prev_vert(N, -1);
    cost[source_node] = 0;
    prev_vert[source_node] = source_node;

    // we initially add the source node and update the costs
    cluster.push_back(source_node);

    // either we take all edges until the end or until the next node stores its edges
    size_t edge_list_end = (source_node + 1 < graph->edges.size()) ? graph->edges[source_node + 1] : graph->destinations.size();

    // for all edges incident from this node
    for(size_t i = graph->edges[source_node]; i < graph->destinations.size() && i < edge_list_end; i++)
    {
        // node that the edge is incident to (s) -> (dest)
        size_t dest_node = graph->destinations[i];
        cost[dest_node] = graph->weights[i]; // the cost is the weight of the edge
        used_edge[dest_node] = i; // we remember the index of the edge we use
        prev_vert[dest_node] = source_node; // we remember the node we came from for this edge
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


                prev_vert[dest_node] = picked_node;// we remember the node we came from for this edge
            }
        }
    } while(cluster.size() < graph->edges.size()); // while there is a vertex that is not reached yet

    std::vector<std::vector<int>> shortest_paths;

    // to find all shortest paths we can just traverse the cluster reversed
    for(int i = cluster.size()-1; i > 0; i--)
    {
        std::vector<int> path;
        int node = cluster[i];
        int prev = prev_vert[node];

        path.push_back(node);

        // as long as the previous node is not the source node we keep iterating
        while(prev != source_node)
        {
            path.push_back(prev);
            node = prev;
            prev = prev_vert[node];
        }

        // for completeness we also add the source node
        path.push_back(prev);
        shortest_paths.push_back(path);
    }

    // as the paths are now reversed, we simply reverse the vectors
    std::reverse(shortest_paths.begin(), shortest_paths.end());
    for(int i = 0; i < shortest_paths.size(); i++)
    {
        std::reverse(shortest_paths[i].begin(), shortest_paths[i].end());
    }

    std::cout << "Debug output:" << std::endl;
    for(int i = 0; i < shortest_paths.size(); i++)
    {
        std::cout << "Paths [" << source_node << "] to [" << shortest_paths[i][shortest_paths[i].size()-1] << "]: ";
        for(int j = 0; j < shortest_paths[i].size(); j++)
        {
            std::cout << " " << shortest_paths[i][j] << ",";
        }
        std::cout << std::endl;
    }

    return shortest_paths;
}