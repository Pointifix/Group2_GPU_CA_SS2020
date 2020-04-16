#include "sssp_sequential.h"

SSSP_Sequential::SSSP_Sequential(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::vector<std::shared_ptr<Path>> SSSP_Sequential::compute(int source_node)
{
    // TODO
    std::vector<int> cluster;

    int N = graph->edges.size();

    while(cluster.size() != graph->edges.size()) // while there is a vertex that is not in the cluster
    {
        std::vector<int> cost(N, 0);

        // For all nodes outside the cluster
        for(int i = 0; i < graph->edges.size(); i++)
        {
            // Skip if the node is in the cluster
            if(std::find(cluster.begin(), cluster.end(), i) != cluster.end())
            {
                continue;
            }

            // Calculate the distance from non-member vertex to s through the cluster
        }

        // select the shortest path and add it to the cluster
        for(;;)
        {

        }
    }

    return std::vector<std::shared_ptr<Path>>();
}