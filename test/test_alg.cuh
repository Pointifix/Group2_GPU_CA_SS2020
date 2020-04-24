#ifndef GROUP2_GPU_CA_SS2020_TEST_CONTOCCUR_H
#define GROUP2_GPU_CA_SS2020_TEST_CONTOCCUR_H

#include "catch.hpp"

#include "alg.cuh"

using namespace alg;

TEST_CASE("Test alg") {

    SECTION("Add") {
        const std::vector<unsigned int> a{1, 2, 3, 4, 5};
        const std::vector<unsigned int> b{10, 8, 6, 4, 2};
        const std::vector<unsigned int> exp{11, 10, 9, 8, 7};
        std::vector<unsigned int> out(exp.size());

        add_parcu(a, b, out);
        CHECK(out == exp);
    }

    SECTION("Count occurrences") {
        const std::vector<unsigned int> a{1u, 1u, 2u, 1u, 3u, 2u, 3u, 1u, 1u, 2u, 0u};
        const std::vector<unsigned int> exp{1u, 5u, 3u, 2u}; // Expected output
        std::vector<unsigned int> out(exp.size());

        SECTION("Sequential") {
            countoccur_seq(a, out);
            CHECK(out == exp);
        }

        SECTION("Parallel") {
            countoccur_parcu(a, out);
            CHECK(out == exp);
        }
    }
}

#endif //GROUP2_GPU_CA_SS2020_TEST_CONTOCCUR_H
