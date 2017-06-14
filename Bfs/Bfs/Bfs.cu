
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include "Bfs.cuh"

__global__ void bfs_kernel(unsigned int* current_set, unsigned int* new_set,
	int current_set_size, int* current_set_size_new,
	Node* node_list, Edge* edge_list, int* color, int* cost, int level)
{
	// get tread number
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = tid; j<current_set_size; j += blockDim.x*gridDim.x) {
		unsigned int index = current_set[j];// get one from the current set
		current_set[j] = 0;                 // make it not visited
		cost[index] = level;
		Node cur_node = node_list[index];
		// all adjacent vertices
		for (int i = cur_node.start; i < cur_node.start + cur_node.edge_num; i++)
		{
			unsigned int id = edge_list[i].dest;
			int old_color = atomicExch((int*)&color[id], BLACK);// visit adjacent node
			if (old_color == WHITE) { // was not visited
				int write_position = atomicAdd((int*) &(*current_set_size_new), 1);
				new_set[write_position] = id; // add to set of the next level
			}
		}
	}
}

void callBFSKernel(const unsigned int blocks,
	const unsigned int threadsPerBlock,
	unsigned int* current_set, unsigned int* new_set,
	int current_set_size, int* current_set_size_new,
	Node* node_list, Edge* edge_list, int* color, int* cost, int level)
{
	bfs_kernel << <blocks, threadsPerBlock >> > (current_set, new_set, current_set_size,
		current_set_size_new, node_list, edge_list, color, cost, level);
}

