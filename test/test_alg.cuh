#ifndef GROUP2_GPU_CA_SS2020_TEST_CONTOCCUR_H
#define GROUP2_GPU_CA_SS2020_TEST_CONTOCCUR_H

#include "catch.hpp"

#include "alg.cuh"

using namespace alg;

TEST_CASE("Test alg") {

    SECTION("Fill") {
        std::vector<uint> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        fill_parcu(a, 69u, 0);
        CHECK(a == std::vector<uint>{69, 69, 69, 69, 69, 69, 69, 69, 69, 69});

        fill_parcu(a, 420u, -39);
        CHECK(a == std::vector<uint>{420, 381, 342, 303, 264, 225, 186, 147, 108, 69});
    }

    SECTION("Add") {
        std::vector<uint> a{1, 2, 3, 4, 5};
        std::vector<uint> b{10, 8, 6, 4, 2};
        std::vector<uint> exp{11, 10, 9, 8, 7};
        std::vector<uint> out(exp.size());

        SECTION("Sequential") {
            add_seq(a.data(), 5, 300);
            CHECK(a == std::vector<uint>{301, 302, 303, 304, 305});
        }

        SECTION("Parallel") {
            add_parcu(a, b, out);
            CHECK(out == exp);
        }
    }

    SECTION("Count occurrences") {
         std::vector<uint> a{1u, 1u, 2u, 1u, 3u, 2u, 3u, 1u, 1u, 2u, 0u};
         std::vector<uint> expa{1u, 5u, 3u, 2u}; // Expected output
        std::vector<uint> outa(expa.size());

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
        std::vector<uint> a_size0{};
        std::vector<uint> exp_size0{};
        std::vector<uint> out_size0{};

        std::vector<uint> a_size1{0xBAD};
        std::vector<uint> exp_size1{0};
        std::vector<uint> out_size1(1);

        std::vector<uint> a_size2{0xCAFE, 0xB00B};
        std::vector<uint> exp_size2{0, 0xCAFE};
        std::vector<uint> out_size2(2);

        std::vector<uint> a_size9{2, 0, 0, 3, 1337, 42, 69, 0, 0xDEADBEEF};
        std::vector<uint> exp_size9{0, 2, 2, 2, 5, 1342, 1384, 1453, 1453};
        std::vector<uint> out_size9(9);
        
        std::vector<uint> a_size256(256);
        fill_parcu(a_size256, 1);
        std::vector<uint> exp_size256(256);
        fill_parcu(exp_size256, 0, 1);
        std::vector<uint> out_size256(256);

        std::vector<uint> a_size1055(1055);
        fill_parcu(a_size1055, 1);
        std::vector<uint> exp_size1055(1055);
        fill_parcu(exp_size1055, 0, 1);
        std::vector<uint> out_size1055(1055);

        SECTION("Sequential") {
            exscan_seq(a_size0, out_size0);
            CHECK(out_size0 == exp_size0);

            exscan_seq(a_size1, out_size1);
            CHECK(out_size1 == exp_size1);

            exscan_seq(a_size2, out_size2);
            CHECK(out_size2 == exp_size2);

            exscan_seq(a_size9, out_size9);
            CHECK(out_size9 == exp_size9);

            std::vector<uint> a_offset {100, 100, 100, 1, 1, 1};
            std::vector<uint> out_offset(6);
            exscan_seq(a_offset, out_offset, 3);
            CHECK(out_offset == std::vector<uint>{0, 0, 0, 0, 1, 2});
        }

        SECTION("Parallel") {
            exscan_parcu(a_size0, out_size0);
            CHECK(out_size0 == exp_size0);

            exscan_parcu(a_size1, out_size1);
            CHECK(out_size1 == exp_size1);

            exscan_parcu(a_size2, out_size2);
            CHECK(out_size2 == exp_size2);

            exscan_parcu(a_size9, out_size9);
            CHECK(out_size9 == exp_size9);

            exscan_parcu(a_size256, out_size256);
            CHECK(out_size256 == exp_size256);

            exscan_parcu(a_size1055, out_size1055);
            CHECK(out_size1055 == exp_size1055);
        }
    }
}

#endif //GROUP2_GPU_CA_SS2020_TEST_CONTOCCUR_H
