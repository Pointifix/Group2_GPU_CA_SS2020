#ifndef SSSP_H_H
#define SSSP_H_H

#include <memory>
#include "graph.h"

class SSSP {
public:
    /**
     * Creates a SSSP instance.
     * @param graph graph that is used for computations.
     */
    explicit SSSP(std::shared_ptr<Graph> graph) : graph(std::move(graph)) {};
    virtual std::shared_ptr<Paths> compute(int source_node) = 0;
protected:
    std::shared_ptr<Graph> graph;
};

#endif /* SSSP_H */