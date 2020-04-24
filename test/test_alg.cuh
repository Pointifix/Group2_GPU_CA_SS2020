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
        const std::vector<unsigned int> expa{1u, 5u, 3u, 2u}; // Expected output
        std::vector<unsigned int> outa(expa.size());

        std::vector<uint> b(1000);
        for (int i = 0; i < b.size(); i++) {
            b[i] = i;
        }
        std::vector<uint> expb(1000);
        std::fill(expb.begin(), expb.end(), 1);
        std::vector<uint> outb(expb.size());

        SECTION("Sequential") {
            countoccur_seq(a, outa);
            CHECK(outa == expa);
        }

        SECTION("Parallel") {
            countoccur_parcu(a, outa);
            CHECK(outa == expa);

            countoccur_parcu(b, outb);
            CHECK(outb == expb);
        }
    }

    SECTION("Prefix sum") {
        const std::vector<unsigned int> a{2, 0, 0, 3, 1337, 42, 69};
        const std::vector<unsigned int> exp{0, 2, 2, 2, 5, 1342, 1384};
        std::vector<unsigned int> out(exp.size());

        SECTION("Sequential") {
            exscan_seq(a, out);
            CHECK(out == exp);
        }

        SECTION("Parallel") {
            exscan_parcu(a, out);
            CHECK(out == exp);
        }
    }
}

#endif //GROUP2_GPU_CA_SS2020_TEST_CONTOCCUR_H
