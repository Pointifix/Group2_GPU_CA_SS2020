#ifndef SSP_Pinned_Memory_H
#define SSP_Pinned_Memory_H

#include "sssp.h"
#include <common.cuh>
#include <alg.cuh>
#include <graph.h>
#include <algorithm>
#include <vector>

class SSSP_Pinned_Memory : public SSSP {
public:
    /**
     * SSSP instance using pinned memory for CUDA computation.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Pinned_Memory(std::shared_ptr<Graph> graph);
    std::shared_ptr<Paths> compute(int source_node) override;
};

#endif /* SSP_Pinned_Memory_H */
