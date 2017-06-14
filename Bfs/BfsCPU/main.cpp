#include <iostream>
#include <list>
#include <stdio.h>

using namespace std;

class Graph
{
	int V;    // No. of vertices
	list<int> *adj;    // Pointer to an array containing adjacency lists
public:
	Graph(int V);  // Constructor
	void addEdge(int v, int w); // function to add an edge to graph
	void BFS(int s);  // prints BFS traversal from a given source s
};

Graph::Graph(int V)
{
	this->V = V;
	adj = new list<int>[V];
}

void Graph::addEdge(int v, int w)
{
	adj[v].push_back(w); // Add w to v’s list.
}

void Graph::BFS(int s)
{
	// Mark all the vertices as not visited
	bool *visited = new bool[V];
	for (int i = 0; i < V; i++)
		visited[i] = false;

	// Create a queue for BFS
	list<int> queue;

	// Mark the current node as visited and enqueue it
	visited[s] = true;
	queue.push_back(s);

	// 'i' will be used to get all adjacent vertices of a vertex
	list<int>::iterator i;

	while (!queue.empty())
	{
		// Dequeue a vertex from queue and print it
		s = queue.front();
		//cout << s << " ";
		queue.pop_front();

		// Get all adjacent vertices of the dequeued vertex s
		// If a adjacent has not been visited, then mark it visited
		// and enqueue it
		for (i = adj[s].begin(); i != adj[s].end(); ++i)
		{
			if (!visited[*i])
			{
				visited[*i] = true;
				queue.push_back(*i);
			}
		}
	}
}

int main()
{
	FILE* fp = fopen("./com-youtube.ungraph.txt", "r");
	if (!fp) {
		printf("file cannot be opened.\n");
		return -1;
	}

	int num_of_nodes;
	int num_of_edges;
	fscanf(fp, "%d", &num_of_nodes);
	fscanf(fp, "%d", &num_of_edges);

	Graph g(num_of_nodes);
	for (int i = 0; i < num_of_edges; i++) {
		int start;
		int end;
		fscanf(fp, "%d %d", &start, &end);
		g.addEdge(start, end);
	}

	g.BFS(2);

	return 0;
}