#include <iostream>

#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"
#include "sssp_sequential.h"

int main()
{
	//std::shared_ptr<Graph> graph = graphgen::generateConnectedGraph(5, 0.2);
	//std::cout << graph->toString();
	//graphio::writeGraph("../output/graph", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("../output/graph");
    std::cout << graph2->toString();

    SSSP_Sequential sequ(graph2);
    std::vector<std::vector<int>> paths = sequ.compute(4);

    graphio::writePaths("../output/path", graph2, paths);
}