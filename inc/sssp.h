#pragma once

#include <c++/4.9/memory>
#include "graph.h"

class SSSP {
public:
    /**
     * Creates a SSSP instance.
     * @param graph graph that is used for computations.
     */
    explicit SSSP(std::shared_ptr<Graph> graph) : graph(std::move(graph)) {};
    virtual std::vector<std::shared_ptr<Path>> compute(int source_node) = 0;
protected:
    std::shared_ptr<Graph> graph;
};