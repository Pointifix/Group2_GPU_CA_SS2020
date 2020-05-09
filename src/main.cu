#include <iostream>

#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"
#include "sssp_sequential.h"
#include "sssp_standard.h"

int main()
{
	std::shared_ptr<Graph> graph = graphgen::generateConnectedGraph(10, 0.3);
	std::cout << graph->toString() << std::endl;
	graphio::writeGraph("output/graph", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("output/graph");
    std::cout << graph2->toString() << std::endl;
    
    SSSP_Sequential sequ(graph2);
    std::shared_ptr<Paths> paths1 = sequ.compute(0);
    std::cout << paths1->toString() << std::endl;

    graphio::writePaths("output/path", paths1);

    SSSP_Standard standard(graph2);
    std::shared_ptr<Paths> paths2 = standard.compute(0);
    std::cout << paths2->toString() << std::endl;

    std::vector<int> path = paths1->getPath(5);

    for(int i = 0; i < path.size(); i++)
    {
        std::cout << path[i] << ", ";
    }
}