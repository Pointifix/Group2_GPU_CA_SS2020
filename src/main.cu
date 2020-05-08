#include <iostream>

#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"
#include "sssp_sequential.h"
#include "sssp_standard.h"

int main()
{
	std::shared_ptr<Graph> graph = graphgen::generateConnectedGraph(200, 0.003);
	std::cout << graph->toString();
	graphio::writeGraph("../output/graph", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("../output/graph-assignment");
    std::cout << graph2->toString();

    SSSP_Sequential sequ(graph2);
    std::vector<std::vector<int>> paths = sequ.compute(0);

    graphio::writePaths("../output/path", graph2, paths);

    SSSP_Standard standard(graph2);
    paths = standard.compute(0);
}