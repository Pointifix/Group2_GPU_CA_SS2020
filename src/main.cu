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
#include "sssp_pinned_memory.h"
#include "sssp_zero_copy_memory.h"
#include "sssp_gpu_search.h"

#ifdef  DEBUG
void test();
#endif

int main()
{
    // Enable Zero Copy
    // Source: https://arrayfire.com/zero-copy-on-tegra-k1/
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.canMapHostMemory) {
        M_RUNTIME_ERROR("Zero copy memory not supported by the GPU");
    }
    cudaSetDeviceFlags(cudaDeviceMapHost);

#ifdef DEBUG
    test();
    //return 0;
#endif

    srand(time(nullptr));

    for (int i = 1; i <= 8; i++)
    {
        int nodes = pow(10, i);

        time_measurement::startMeasurement("Graph Generation");
        std::shared_ptr<Graph> graph = graphgen::generateGraph(nodes, graphgen::calculateDensity(nodes * 5, nodes, true));
        time_measurement::endMeasurement("Graph Generation");

        //std::cout << graph->toString() << std::endl;

        /*
        time_measurement::startMeasurement("Graph Output");
        graphio::writeGraph("output/graph", graph);
        time_measurement::endMeasurement("Graph Output");

        time_measurement::startMeasurement("Graph Input");
        std::shared_ptr<Graph> graph2 = graphio::readGraph("output/graph");
        time_measurement::endMeasurement("Graph Input");
         */


        int random_source = rand() % nodes;

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
        std::shared_ptr<Paths> paths3 = thrust.compute(random_source);
        time_measurement::endMeasurement("SSSP Thrust");

        SSSP_Pinned_Memory pinned(graph);
        time_measurement::startMeasurement("SSSP Pinned");
        std::shared_ptr<Paths> paths4 = pinned.compute(random_source);
        time_measurement::endMeasurement("SSSP Pinned");

        SSSP_Zero_Copy_Memory zeroCopy(graph);
        time_measurement::startMeasurement("SSSP Zero Copy");
        std::shared_ptr<Paths> paths5 = zeroCopy.compute(random_source);
        time_measurement::endMeasurement("SSSP Zero Copy");

        SSSP_GPU_Search gpuSearch(graph);
        time_measurement::startMeasurement("SSSP GPU Search");
        std::shared_ptr<Paths> paths6 = gpuSearch.compute(random_source);
        time_measurement::endMeasurement("SSSP GPU Search");

        std::cout << "path 1 and 2 same? " << paths1->isEqualTo(paths2.get()) << std::endl;
        std::cout << "path 2 and 3 same? " << paths2->isEqualTo(paths3.get()) << std::endl;
        std::cout << "path 1 and 3 same? " << paths1->isEqualTo(paths3.get()) << std::endl;
        std::cout << "path 1 and 4 same? " << paths1->isEqualTo(paths4.get()) << std::endl;
        std::cout << "path 1 and 5 same? " << paths1->isEqualTo(paths5.get()) << std::endl;
        std::cout << "path 1 and 6 same? " << paths1->isEqualTo(paths6.get()) << std::endl;

        std::cout << "\nGraph (" << graph->edges.size() << " Vertices, "<< graph->destinations.size() << " Edges)" << std::endl;
    }
    time_measurement::printMeasurements();
}

#ifdef DEBUG
void test() {
    M_A(true); // Use M_A to make assertions!
}
#endif