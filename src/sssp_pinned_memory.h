#pragma once
#include "sssp.h"

class SSSP_Pinned_Memory : public SSSP {
public:
    /**
     * SSSP instance using pinned memory for CUDA computation.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Pinned_Memory(std::shared_ptr<Graph> graph);
    std::vector<std::shared_ptr<Path>> compute(int source_node) override;
};
