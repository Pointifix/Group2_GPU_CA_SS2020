#include <iostream>
#include <chrono>

#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"
#include "sssp_sequential.h"
#include "sssp_standard.h"
#include "sssp_thrust.h"

int main()
{
    srand(time(nullptr));

	std::shared_ptr<Graph> graph = graphgen::generateGraph(1'000, 0.01);
	std::cout << graph->toString();
	graphio::writeGraph("output/graph", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("output/graph");
    std::cout << graph2->toString();

    //auto adjacencyMatrix = graph2->getAdjacencyMatrix();

    /*
    std::cout << "Adjacency Matrix:" << std::endl;
    for(int i = 0; i < graph2->edges.size(); i++)
    {
        for(int j = 0; j < graph2->edges.size(); j++)
        {
            std::cout << adjacencyMatrix[i][j] << ", ";
        }
        std::cout << std::endl;
    }
     */

    int random_source = rand() % graph2->edges.size();

    SSSP_Sequential sequ(graph2);
    std::shared_ptr<Paths> paths1 = sequ.compute(random_source);
    std::cout << paths1->toString() << std::endl;

    graphio::writePaths("output/path_sequential", paths1);

    std::cout << "Standard SSSP" << std::endl;
    SSSP_Standard standard(graph2);
    std::shared_ptr<Paths> paths2 = standard.compute(random_source);
    std::cout << paths2->toString() << std::endl;

    graphio::writePaths("output/path_standard", paths2);

    SSSP_Thrust thrust(graph2);
    std::shared_ptr<Paths> paths3 = standard.compute(random_source);
    std::cout << paths3->toString() << std::endl;

    std::cout << "path 1 and 2 same? " << paths1->isEqualTo(paths2.get()) << std::endl;
    std::cout << "path 2 and 3 same? " << paths2->isEqualTo(paths3.get()) << std::endl;
    std::cout << "path 1 and 3 same? " << paths1->isEqualTo(paths3.get()) << std::endl;
}