#pragma once
#include "sssp.h"

class SSSP_Sequential : public SSSP {
public:
    /**
     * SSSP instance for sequential computation the CPU.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Sequential(std::shared_ptr<Graph> graph);
};
