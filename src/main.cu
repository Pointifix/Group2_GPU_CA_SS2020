#include <iostream>
#include <chrono>

#include "alg.cuh"
#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"
#include "sssp_sequential.h"
#include "sssp_standard.h"
#include "sssp_thrust.h"
#include "time_measurement.h"

int main()
{
    srand(time(nullptr));

    for (int i = 1; i <= 5; i++)
    {
        int nodes = pow(10, i);

        time_measurement::startMeasurement("Graph Generation");
        std::shared_ptr<Graph> graph = graphgen::generateGraph(nodes, graphgen::calculateDensity(nodes * 5, nodes, true));
        time_measurement::endMeasurement("Graph Generation");

        /*
        time_measurement::startMeasurement("Graph Output");
        graphio::writeGraph("output/graph", graph);
        time_measurement::endMeasurement("Graph Output");

        time_measurement::startMeasurement("Graph Input");
        std::shared_ptr<Graph> graph2 = graphio::readGraph("output/graph");
        time_measurement::endMeasurement("Graph Input");
         */

        int random_source = rand() % graph->edges.size();

        SSSP_Sequential sequ(graph);
        time_measurement::startMeasurement("SSSP Sequential");
        std::shared_ptr<Paths> paths1 = sequ.compute(random_source);
        time_measurement::endMeasurement("SSSP Sequential");

        //graphio::writePaths("output/path_sequential", paths1);

        SSSP_Standard standard(graph);
        time_measurement::startMeasurement("SSSP Standard");
        std::shared_ptr<Paths> paths2 = standard.compute(random_source);
        time_measurement::endMeasurement("SSSP Standard");

        //graphio::writePaths("output/path_standard", paths2);

        SSSP_Thrust thrust(graph);
        time_measurement::startMeasurement("SSSP Thrust");
        std::shared_ptr<Paths> paths3 = standard.compute(random_source);
        time_measurement::endMeasurement("SSSP Thrust");

        std::cout << "path 1 and 2 same? " << paths1->isEqualTo(paths2.get()) << std::endl;
        std::cout << "path 2 and 3 same? " << paths2->isEqualTo(paths3.get()) << std::endl;
        std::cout << "path 1 and 3 same? " << paths1->isEqualTo(paths3.get()) << std::endl;

        std::cout << "\nGraph (" << graph->edges.size() << " Vertices, "<< graph->destinations.size() << " Edges)" << std::endl;
    }
    time_measurement::printMeasurements();
}