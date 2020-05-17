#include "sssp_sequential.h"

SSSP_Sequential::SSSP_Sequential(std::shared_ptr<Graph> graph) : SSSP(std::move(graph)) {
}

std::shared_ptr<Paths> SSSP_Sequential::compute(int source_node)
{
    std::vector<weight_t> cost(graph->edges.size(), std::numeric_limits<int>::max());
    std::vector<pos_t> previous_nodes(graph->edges.size(), M_INVALID_POSITION);
    std::vector<mask_t> mask(graph->edges.size(), M_MASK_FALSE);

    mask.at(source_node) = M_MASK_TRUE;
    cost.at(source_node) = 0;

    while (std::find(mask.begin(), mask.end(), M_MASK_TRUE) != mask.end())
    {
        for (int i = 0; i < mask.size(); i++)
        {
            if (mask[i] == 1)
            {
                pos_t first = graph->edges[i];
                pos_t last = (i + 1 < graph->edges.size()) ? graph->edges[i + 1] : graph->destinations.size();

                for (int j = first; j < last; j++)
                {
                    int new_cost = cost[i] + graph->weights[j];

                    if (new_cost < cost[graph->destinations[j]])
                    {
                        cost[graph->destinations[j]] = new_cost;
                        previous_nodes[graph->destinations[j]] = i;
                        mask[graph->destinations[j]] = M_MASK_TRUE;
                    }
                }
                mask[i] = M_MASK_FALSE;
            }
        }
    }

    return std::make_shared<Paths>(Paths(previous_nodes, cost, source_node, graph));
}