#include <iostream>

#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"

int main()
{
	std::shared_ptr<Graph> graph = graphgen::generateConnectedGraph(500, 0.7);

	std::cout << graph->to_string();

	graphio::writeGraph("graph.txt", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("graph.txt");

    if (graph2 == nullptr) std::cout << "nullptr" << std::endl;

    std::cout << graph2->to_string();
}