#pragma once

#include "catch.hpp"

#include "graph_generator.h"

#include <cstdlib>

using namespace graphgen;

constexpr float den(int e, int v, bool directed) {
    return (float)(directed ? 1 : 2) * (float)e / ((float)v * ((float)v - 1));
}

TEST_CASE("Graph Generator simple constraints") {
    CHECK_FALSE(generateConnectedGraph(-1, 1));
    CHECK_FALSE(generateConnectedGraph(0, 1.0001));
    CHECK_FALSE(generateConnectedGraph(1, -0.0001));
    CHECK(generateConnectedGraph(2, 0));
    CHECK(generateConnectedGraph(3, 0.5));
    CHECK(generateConnectedGraph(4, 1));
    CHECK(generateConnectedGraph(5, 0.25, true));
    CHECK(generateConnectedGraph(6, 0.75, false));
}

TEST_CASE("Graph Generator simple graph") {
    std::shared_ptr<Graph> g;

    g = generateConnectedGraph(1000, 0);
    CHECK(g->edges.size() == 1000);
    CHECK(g->destinations.size() == 1000 - 1);
    CHECK(g->weights.size() == 1000 - 1);

    g = generateConnectedGraph(10, den(5, 10, true), true);
    CHECK(g->edges.size() == 10);
    CHECK(g->destinations.size() == 5);
    CHECK(g->weights.size() == 5);

    g = generateConnectedGraph(10, den(5, 10, false), false);
    CHECK(g->edges.size() == 10);
    CHECK(g->destinations.size() == 5);
    CHECK(g->weights.size() == 5);

    g = generateConnectedGraph(1000, den(700, 1000, true), true);
    CHECK(g->edges.size() == 1000);
    CHECK(g->destinations.size() == 700);
    CHECK(g->weights.size() == 700);

    g = generateConnectedGraph(1000, den(300, 1000, false), false);
    CHECK(g->edges.size() == 1000);
    CHECK(g->destinations.size() == 300);
    CHECK(g->weights.size() == 300);
}

TEST_CASE("Graph Generator weak connectedness") {
    std::shared_ptr<Graph> g;
    std::vector<bool> weakly_connected(1000);

    int num_nodes_a[] {1000};
    float density_a[] {0, 0.5, 1};
    int directed_a[] {true, false};

    for (int num_nodes : num_nodes_a) {
        for (float density : density_a) {
            for (bool directed : directed_a) {
                std::fill(weakly_connected.begin(), weakly_connected.end(), false);
                int weakly_connected_count = 0;
                g = generateConnectedGraph(num_nodes, density, directed);

                auto section_name = "num_nodes=" + std::to_string(num_nodes) +
                                   " density=" + std::to_string(density) +
                                   " directed=" + std::to_string(directed);
                SECTION(section_name) {
                    REQUIRE(g->edges.size() == num_nodes);

                    for (int nodei = 1; nodei < num_nodes; nodei++) {
                        const int start = g->edges[nodei - 1];

                        if (start == num_nodes) {
                            // If the current node has no outgoing edges, continue
                            continue;
                        } else if (!weakly_connected[nodei]) {
                            // If the current node has outgoing edges, it is weakly connected for sure!
                            weakly_connected[nodei] = true;
                            weakly_connected_count++;
                        }

                        // Now go through all outgoing edges of the current node
                        const int end = g->edges[nodei];
                        for (int edgei = start; edgei < end; edgei++) {
                            int desti = g->destinations[edgei];
                            if (!weakly_connected[desti]) {
                                // For each outgoing edge of the current node,
                                // mark the connected node as weakly connected
                                weakly_connected[desti] = true;
                                weakly_connected_count++;
                            }
                        }
                    }

                    CHECK(weakly_connected_count == num_nodes);
                }
            }
        }
    }
}