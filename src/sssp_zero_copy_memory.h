#pragma once
#include "sssp.h"

class SSSP_Zero_Copy_Memory : public SSSP {
    /**
     * Instance of a SSSP for computation using zero copy memory on CUDA.
     * @param graph graph that is used for computations.
     */
    explicit SSSP_Zero_Copy_Memory(std::shared_ptr<Graph> graph);
};