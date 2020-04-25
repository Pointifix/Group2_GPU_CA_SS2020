#include <iostream>

#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"
#include "sssp_sequential.h"

int main()
{
	//std::shared_ptr<Graph> graph = graphgen::generateConnectedGraph(5, 0.2);
	//std::cout << graph->to_string();
	//graphio::writeGraph("graph.txt", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("graph.txt");
    std::cout << graph2->to_string();

    SSSP_Sequential sequ(graph2);
    sequ.compute(4);


}