#ifndef SSSP_Zero_Copy_Memory_H
#define SSSP_Zero_Copy_Memory_H

#include "sssp.h"

class SSSP_Zero_Copy_Memory : public SSSP {
    /**
     * Instance of a SSSP for computation using zero copy memory on CUDA.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Zero_Copy_Memory(std::shared_ptr<Graph> graph);
    std::shared_ptr<Paths> compute(int source_node) override;
};

#endif /* SSSP_Zero_Copy_Memory_H */