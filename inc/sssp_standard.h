#ifndef SSSP_Standard_H
#define SSSP_Standard_H

#include "sssp.h"
#include <iostream>

class SSSP_Standard : public SSSP {
public:
    /**
     * SSSP instance for standard CUDA computation.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Standard(std::shared_ptr<Graph> graph);
    std::vector<std::vector<int>> compute(int source_node) override;
};

#endif /* SSSP_Standard_H */
