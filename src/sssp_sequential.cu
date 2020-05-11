#include "sssp_sequential.h"

SSSP_Sequential::SSSP_Sequential(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::shared_ptr<Paths> SSSP_Sequential::compute(int source_node)
{
    std::vector<int> cost(graph->edges.size(), std::numeric_limits<int>::max());
    std::vector<int> previous_nodes(graph->edges.size(), -1);
    std::vector<int> mask(graph->edges.size(), 0);

    mask.at(source_node) = 1;
    cost.at(source_node) = 0;

    while (std::find(mask.begin(), mask.end(), true) != mask.end())
    {
        for (int i = 0; i < mask.size(); i++)
        {
            if (mask[i] == 1)
            {
                int first = graph->edges[i];
                int last = (i + 1 < graph->edges.size()) ? graph->edges[i + 1] : graph->destinations.size();

                for (int j = first; j < last; j++)
                {
                    int new_cost = cost[i] + graph->weights[j];

                    if (new_cost < cost[graph->destinations[j]])
                    {
                        cost[graph->destinations[j]] = new_cost;
                        previous_nodes[graph->destinations[j]] = i;
                        mask[graph->destinations[j]] = 1;
                    }
                }
                mask[i] = 0;
            }
        }
    }

    return std::make_shared<Paths>(Paths(previous_nodes, cost, source_node, graph));
}