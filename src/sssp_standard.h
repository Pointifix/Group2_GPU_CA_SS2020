#pragma once
#include "sssp.h"

class SSSP_Standard : public SSSP {
public:
    /**
     * SSSP instance for standard CUDA computation.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Standard(std::shared_ptr<Graph> graph);
};
