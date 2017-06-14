#ifndef BFS_CUH
#define BFS_CUH

#define WHITE	0				/* not visited */
#define GREY	1				/* visiting */
#define BLACK	2				/* visited */
#define INF		2147483647		/* infinity distance */
#define MAX_LEVEL		20000	
#define THREAD_PER_BLOCK 512    /* can depend on number of nodes*/
#define BLOCK_NUM 256

typedef struct node_t
{
	unsigned int start;		/* starting index of edges */
	unsigned int edge_num;
} Node;

typedef struct edge_t
{
	unsigned int dest;			/* index of nodes */
	unsigned int cost;
} Edge;

void callBFSKernel(const unsigned int blocks,
	const unsigned int threadsPerBlock,
	unsigned int* current_set, unsigned int* new_set,
	int current_set_size, int* current_set_size_new,
	Node* node_list, Edge* edge_list, int* color, int* cost, int level);

#endif // BFS_CUH