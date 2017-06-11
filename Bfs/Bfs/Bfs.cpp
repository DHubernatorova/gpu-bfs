#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Bfs.cuh"
#include <cuda_runtime.h>

void printLevels(int* counter, int num_of_nodes, unsigned int source_node_no,
	unsigned int num_of_edges)
{
	FILE* fp = fopen("./output.txt", "w");
	if (!fp) {
		printf("log file cannot be opened.\n");
		return;
	}

	fprintf(fp, "************** graph **************\n");
	fprintf(fp, "there are %d nodes in the graph.\n", num_of_nodes);
	fprintf(fp, "the source node is %u.\n", source_node_no);
	fprintf(fp, "there are %u edges in the graph.\n", num_of_edges);

	int num_of_levels;
	for (int i = 1; i<MAX_LEVEL; i++) {
		if (counter[i] == 0) {
			num_of_levels = i - 1;
			fprintf(fp, "The num of levels is %d\n", num_of_levels);
			break;
		}
		fprintf(fp, "%d\n", counter[i]);
	}

	fclose(fp);
}

int main(int argc, char** argv)
{
	// host
	unsigned int num_of_nodes;
	unsigned int num_of_edges;
	unsigned int source_node_no;

	Node* node_list;
	Edge* edge_list;
	int* color;
	int* cost;
	int* counter;

	bool* visited;
	bool* mask;
	bool* updating_mask;

	unsigned int* current_set;
	int* current_set_size_new;

	// device
	Node* d_node_list;
	Edge* d_edge_list;
	int* d_color;
	int* d_cost;
	int* d_counter;
	unsigned int* d_current_set_a;
	unsigned int* d_current_set_b;
	int* d_current_set_size_new;

	if (argc != 2) {
		printf("please give the path of the file with graph.\n");
		return -1;
	}

	// reading from file
	FILE* fp = fopen(argv[1], "r");
	if (!fp) {
		printf("Cannot open the graph file.\n");
		return -1;
	}


	fscanf(fp, "%d", &num_of_nodes);

	node_list = (Node*)malloc(sizeof(Node) * num_of_nodes);
	color = (int*)malloc(sizeof(int) * num_of_nodes);
	cost = (int*)malloc(sizeof(int) * num_of_nodes);
	visited = (bool*)malloc(sizeof(int) * num_of_nodes);
	mask = (bool*)malloc(sizeof(bool) * num_of_nodes);
	updating_mask = (bool*)malloc(sizeof(bool) * num_of_nodes);

	// initialize
	unsigned int start;
	unsigned int edge_num;
	for (int i = 0; i<num_of_nodes; i++) {
		fscanf(fp, "%u %u", &start, &edge_num);
		node_list[i].start = start;
		node_list[i].edge_num = edge_num;
		color[i] = WHITE;
		cost[i] = INF;
		visited[i] = false;
		mask[i] = false;
		updating_mask[i] = false;


	}

	fscanf(fp, "%u", &source_node_no);

	fscanf(fp, "%u", &num_of_edges);

	edge_list = (Edge*)malloc(sizeof(Edge) * num_of_edges);

	unsigned int dest;
	unsigned int currentCost;
	for (int i = 0; i<num_of_edges; i++) {
		fscanf(fp, "%u %u", &dest, &currentCost);
		edge_list[i].dest = dest;
		edge_list[i].cost = currentCost;
	}

	counter = (int*)malloc(sizeof(int) * MAX_LEVEL);
	for (int i = 0; i<MAX_LEVEL; i++) {
		counter[i] = 0;
	}

	fclose(fp);

	// allocation
	current_set = (unsigned int*)malloc(sizeof(unsigned int) * num_of_nodes);
	for (int i = 0; i<num_of_nodes; i++) {
		current_set[i] = INF;
	}
	current_set_size_new = (int*)malloc(sizeof(int));
	*current_set_size_new = 0;

	// device allocation and copy
	cudaMalloc((void**)&d_node_list, sizeof(Node) * num_of_nodes);
	cudaMemcpy(d_node_list, node_list, sizeof(Node) * num_of_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_edge_list, sizeof(Edge) * num_of_edges);
	cudaMemcpy(d_edge_list, edge_list, sizeof(Edge) * num_of_edges, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_color, sizeof(int) * num_of_nodes);
	cudaMemcpy(d_color, color, sizeof(int) * num_of_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_cost, sizeof(int) * num_of_nodes);
	cudaMemcpy(d_cost, cost, sizeof(int) * num_of_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_counter, sizeof(int) * MAX_LEVEL);
	cudaMemcpy(d_counter, counter, sizeof(int) * MAX_LEVEL, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_current_set_a, sizeof(unsigned int) * num_of_nodes);
	cudaMemcpy(d_current_set_a, current_set, sizeof(unsigned int) * num_of_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_current_set_b, sizeof(unsigned int) * num_of_nodes);
	cudaMemcpy(d_current_set_b, current_set, sizeof(unsigned int) * num_of_nodes, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_current_set_size_new, sizeof(int) * 1);
	cudaMemcpy(d_current_set_size_new, current_set_size_new, sizeof(int) * 1, cudaMemcpyHostToDevice);

	// start bfs
	// visiting the source node now(CPU)
	color[source_node_no] = BLACK;
	current_set[0] = source_node_no;
	cost[source_node_no] = 0;

	// synchronize to GPU mem
	cudaMemcpy(d_color, color, sizeof(int) * num_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_current_set_a, current_set, sizeof(unsigned int) * num_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cost, cost, sizeof(int) * num_of_nodes, cudaMemcpyHostToDevice);

	int current_set_size = 1;          // only source node in it     
	int block_num = BLOCK_NUM;
	int thread_num = THREAD_PER_BLOCK;

	int level = 0;                     // used to control the current_set_a/b to visit
	while (current_set_size != 0) {
		if (level % 2 == 0) {
			cudaMemset(d_current_set_size_new, 0, sizeof(int));
			callBFSKernel(block_num, thread_num, d_current_set_a, d_current_set_b, current_set_size, d_current_set_size_new,
				d_node_list, d_edge_list, d_color, d_cost, level);
			cudaThreadSynchronize();
			cudaMemcpy(current_set_size_new, d_current_set_size_new, sizeof(int), cudaMemcpyDeviceToHost);
			current_set_size = *current_set_size_new;

		}
		else {

			cudaMemset(d_current_set_size_new, 0, sizeof(int));
			callBFSKernel(block_num, thread_num, d_current_set_b, d_current_set_a, current_set_size, d_current_set_size_new,
				d_node_list, d_edge_list, d_color, d_cost, level);
			cudaThreadSynchronize();
			cudaMemcpy(current_set_size_new, d_current_set_size_new, sizeof(int), cudaMemcpyDeviceToHost);
			current_set_size = *current_set_size_new;

		}
		level++;
	}

	// copy the result from GPU to CPU mem
	cudaMemcpy(cost, d_cost, sizeof(unsigned int)*num_of_nodes, cudaMemcpyDeviceToHost);

	// calculate counter
	for (int i = 0; i<num_of_nodes; i++) {
		counter[cost[i]] ++;
	}

	printLevels(counter, num_of_nodes, source_node_no, num_of_edges);

	// free device memory
	cudaFree(d_node_list);
	cudaFree(d_edge_list);
	cudaFree(d_cost);
	cudaFree(d_color);
	cudaFree(d_counter);
	cudaFree(d_current_set_a);
	cudaFree(d_current_set_b);
	cudaFree(d_current_set_size_new);

	// free host memory
	free(node_list);
	free(edge_list);
	free(color);
	free(cost);
	free(mask);
	free(visited);
	free(updating_mask);
	free(current_set);
	free(current_set_size_new);

	return 0;
}
