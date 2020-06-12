#include "alg.cuh"

namespace alg {
    __global__ void SSSP_Kernel(const pos_t* edges, const pos_t* destinations, const weight_t * weights,
                                pos_t* previous_node, mask_t* mask, weight_t * cost,
                                size_t nodes_amount, size_t edges_amount)
    {
        uint tid = threadIdx.x + blockDim.x * blockIdx.x;

        if (tid >= nodes_amount) return;

        if (mask[tid])
        {
            pos_t first = edges[tid];
            pos_t last = (tid + 1 < nodes_amount) ? edges[tid + 1] : edges_amount;

            mask[tid] = M_MASK_FALSE;

            for (pos_t i = first; i < last; i++)
            {
                pos_t nid = destinations[i];

                if(cost[nid] > cost[tid] + weights[i])
                {
                    weight_t new_cost = cost[tid] + weights[i];

                    atomicMin(&cost[nid], new_cost);

                    if (cost[nid] == new_cost)
                    {
                        previous_node[nid] = tid;
                        mask[nid] = M_MASK_TRUE;
                    }
                }
            }
        }
    }

    __device__ void selection_sort( int *data, int left, int right )
    {
        for( int i = left ; i <= right ; ++i ){
            int min_val = data[i];
            int min_idx = i;

            // Find the smallest value in the range [left, right].
            for( int j = i+1 ; j <= right ; ++j ){
                int val_j = data[j];
                if( val_j < min_val ){
                    min_idx = j;
                    min_val = val_j;
                }
            }

            // Swap the values.
            if( i != min_idx ){
                data[min_idx] = data[i];
                data[i] = min_val;
            }
        }
    }

    __global__ void setup_kernel(curandState *state, int seed, int num_nodes)
    {
        uint id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= num_nodes) return;

        curand_init(seed, id, 0, &state[id]);
    }

    __global__ void random_graph_Kernel(curandState *state, const pos_t *edges, pos_t *destinations, weight_t *weights, int num_edges, int num_nodes, int max_weight)
    {
        uint id = threadIdx.x + blockIdx.x * blockDim.x;

        if(id >= num_nodes) return;

        int first = edges[id];
        int last = (id + 1 < num_nodes) ? edges[id + 1] : num_edges;

        pos_t random_node;
        curandState local_state = state[id];
        bool already_connected;

        for (int i = first; i < last; i++)
        {
            do {
                already_connected = false;
                random_node = curand_uniform(&local_state) * num_nodes;

                for (int j = first; j < i; j++)
                {
                    if (destinations[j] == random_node)
                    {
                        already_connected = true;
                        break;
                    }
                }
            } while (random_node == id || already_connected);

            destinations[i] = random_node;

            weights[i] = curand_uniform(&local_state) * max_weight;
        }

        selection_sort(destinations, first, last - 1);
    }

    // -----------------------------------------------------------------------------------------------------------------
    // PARALLEL IMPLEMENTATIONS (CUDA)
    // -----------------------------------------------------------------------------------------------------------------

    template<class T> __global__ void _fill_parcu(T *a, size_t N, T value) {
        uint i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < N) {
            a[i] = value;
        }
    }

    template<class T> void fill_parcu(T *d_a, const size_t &Na, const T &value) {
        int threadsPerBlock = M_BLOCKSIZE;
        int numBlocks = (int) ceil((float)Na / (float)threadsPerBlock);
        M_CFUN((_fill_parcu<<< numBlocks, threadsPerBlock >>>(d_a, Na, value)));
    }

    template<class T> void set_parcu(T *d_a, const pos_t &position, const T &value) {
        M_CFUN((_fill_parcu<<< 1, 1 >>>(&d_a[position], 1, value)));
    }

    __device__ bool d_contains;

    template<class T> __global__ void _contains_parcu(const T *a, const size_t Na, const T value) {
        uint i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i >= Na) {
            __syncthreads();
            __syncthreads();
            return;
        }

        uint tid = threadIdx.x;
        __shared__ bool s_contains;

        if (tid == 0) s_contains = false;
        __syncthreads();

        if (i == 0) d_contains = false;
        if (a[i] == value) s_contains = true;

        __syncthreads();
        if (tid == 0 && s_contains) d_contains = true;
    }

    template <class T> void contains_parcu(const T *d_a, const size_t &Na, const T &value, bool *out) {
        size_t remainingElements = Na;
        int numBlocks = 1;
        while(remainingElements > 0) {
            size_t numElements = min((size_t)numBlocks * (size_t)M_BLOCKSIZE, remainingElements);
            size_t blockStart = Na - remainingElements;

            M_CFUN((_contains_parcu<<< numBlocks, M_BLOCKSIZE >>>(&d_a[blockStart], numElements, value)));
            M_C(cudaMemcpyFromSymbol(out, d_contains, sizeof(bool), 0, cudaMemcpyDeviceToHost));
            if (*out) {
                return;
            }

            remainingElements -= numElements;
            numBlocks *= 2;
        }
        *out = false;
    }

    // -----------------------------------------------------------------------------------------------------------------
    // TEMPLATES
    // -----------------------------------------------------------------------------------------------------------------

    template void fill_parcu(bool *d_a, const size_t &Na, const bool &value);
    template void fill_parcu(int *d_a, const size_t &Na, const int &value);

    template void set_parcu(bool *d_a, const pos_t &position, const bool &value);
    template void set_parcu(int *d_a, const pos_t &position, const int &value);

    template void contains_parcu(const bool *d_a, const size_t &Na, const bool &value, bool *out);
}