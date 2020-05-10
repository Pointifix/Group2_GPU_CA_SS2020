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
	std::shared_ptr<Graph> graph = graphgen::generateConnectedGraph(100, 0.05);
	std::cout << graph->toString();
	graphio::writeGraph("graph", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("graph");
    std::cout << graph2->toString();
    
    SSSP_Sequential sequ(graph2);
    std::shared_ptr<Paths> paths1 = sequ.compute(0);
    std::cout << paths1->toString() << std::endl;

    //graphio::writePaths("output/path", graph2, paths);

    SSSP_Standard standard(graph2);
    std::shared_ptr<Paths> paths2 = standard.compute(0);
    std::cout << paths2->toString() << std::endl;

    SSSP_Thrust thrust(graph2);
    std::shared_ptr<Paths> paths3 = standard.compute(0);
    std::cout << paths3->toString() << std::endl;

    std::cout << "path 1 and 2 same? " << paths1->isEqualTo(paths2.get()) << std::endl;
    std::cout << "path 2 and 3 same? " << paths2->isEqualTo(paths3.get()) << std::endl;
    std::cout << "path 1 and 3 same? " << paths1->isEqualTo(paths3.get()) << std::endl;
}