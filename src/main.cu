#include <iostream>

#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"
#include "sssp_sequential.h"
#include "sssp_standard.h"
#include "sssp_thrust.h"

int main()
{
	std::shared_ptr<Graph> graph = graphgen::generateConnectedGraph(200, 0.01);
	std::cout << graph->toString();
	graphio::writeGraph("graph", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("graph");
    std::cout << graph2->toString();
    
    SSSP_Sequential sequ(graph2);
    std::shared_ptr<Paths> paths1 = sequ.compute(0);
    std::cout << paths1->toString() << std::endl;

    //graphio::writePaths("output/path", graph2, paths);

    SSSP_Standard standard(graph2);
    paths1 = standard.compute(0);

    SSSP_Thrust thrust(graph2);
    paths1 = standard.compute(0);
}