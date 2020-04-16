#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

#include "graph.h"
#include "graph_generator.h"
#include "graph_io.h"
#include "sssp.h"

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, int size);
cudaError_t initKernel(int *a, int size);

void printStartAndEndOfArray(int *a, int size, int elements, char name[]);

__device__ int getGlobalIdx_3D_3D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = getGlobalIdx_3D_3D();
    c[i] = a[i] + b[i];
}

__global__ void initKernel(int *a)
{
	int i = getGlobalIdx_3D_3D();
	a[i] = i + 1;
}

int main()
{
	std::shared_ptr<Graph> graph = graphgen::generateConnectedGraph(4, 0.2);

	std::cout << graph->to_string();

	graphio::writeGraph("graph.txt", graph);

    std::shared_ptr<Graph> graph2 = graphio::readGraph("graph.txt");

    if (graph2 == nullptr) std::cout << "nullptr" << std::endl;

    std::cout << graph2->to_string();

	// old test code;
	/*
    const int arraySize = 200000000;
	int* a = new int[arraySize];
	int* b = new int[arraySize];
	int* c = new int[arraySize];
	cudaError_t cudaStatus;

	cout << "array size: " << arraySize << endl;

	cout << "\n---CPU---\n" << endl;

	auto startCPU = chrono::steady_clock::now();
	for (size_t i = 0; i < arraySize; i++)
	{
		a[i] = i;
	}
	auto endCPU = chrono::steady_clock::now();
	cout << "Time Elapsed Init: " << chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU).count() << " ms" << endl;
	startCPU = chrono::steady_clock::now();
	for (size_t i = 0; i < arraySize; i++)
	{
		b[i] = i;
	}
	endCPU = chrono::steady_clock::now();
	cout << "Time Elapsed Init: " << chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU).count()	<< " ms" << endl;

	startCPU = chrono::steady_clock::now();
	for (size_t i = 0; i < arraySize; i++)
	{
		c[i] = a[i] + b[i];
	}
	endCPU = chrono::steady_clock::now();
	cout << "Time Elapsed Add: " << chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU).count() << " ms" << endl;

	cout << "\n---GPU---\n" << endl;
	cudaStatus = initKernel(a, arraySize);
	cudaStatus = initKernel(b, arraySize);

	//printStartAndEndOfArray(a, arraySize, 20, "a");
	//printStartAndEndOfArray(b, arraySize, 20, "b");

    cudaStatus = addWithCuda(c, a, b, arraySize);

	//printStartAndEndOfArray(c, arraySize, 20, "c");

    cudaStatus = cudaDeviceReset();

    return cudaStatus;
	*/
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
    cudaError_t cudaStatus;
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);


    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = ceil((double)size / 1024);
	dim3 threadsPerBlock(32, 32);

	cudaEventRecord(start, 0);
    addKernel<<<numBlocks, threadsPerBlock >>>(dev_c, dev_a, dev_b);
	cudaEventRecord(end, 0);

	cudaDeviceSynchronize();
	cudaEventElapsedTime(&milliseconds, start, end);
	cout << fixed << "Time Elapsed Add: " << milliseconds << "ms" << endl;

    cudaStatus = cudaGetLastError();
    cudaStatus = cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

	return cudaStatus;
}

cudaError_t initKernel(int * a, int size)
{
	int *dev_a = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaStatus = cudaSetDevice(0);

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));

	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = ceil((double)size / 1024);
	dim3 threadsPerBlock(32, 32);

	cudaEventRecord(start, 0);
	initKernel<<<numBlocks, threadsPerBlock>>>(dev_a);
	cudaEventRecord(end, 0);

	cudaDeviceSynchronize();
	cudaEventElapsedTime(&milliseconds, start, end);
	cout << fixed << "Time Elapsed Init: " << milliseconds << "ms" << endl;

	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);

	return cudaStatus;
}

void printStartAndEndOfArray(int *a, int size, int elements, char name[])
{
	int elementsPerSide = elements / 2;

	cout << "Array " << name << endl;
	for (size_t i = 0; i < size; i++)
	{
		if (i == elementsPerSide)
		{
			i = size - elementsPerSide;
			cout << "\n..." << endl;
		}
		cout << a[i] << ", \t";
	}
	cout << "\n" << endl;
}