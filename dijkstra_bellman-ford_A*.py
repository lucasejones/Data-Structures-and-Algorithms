
# learn about iterative deepening and how to implement it. lots of hype around this algo because it combines the time efficiency of bfs with the space efficiency of dfs. 
	# done! 

# learn about negative weights (bellman ford) and A*.
	# working on now 

# then finally you can get back to the dynamic programming card on leetcode, having gone through greedy algos, which contained activity selection, graph coloring, and dijkstra's, which prompted review of bfs and dfs as they apply outside of trees, and bellman ford and A*.





# given a weighted, directed graph (but can work with undirected as well):

# (u, v, w) represent edge from vertex `u` to vertex `v` having weight `w`
edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
        (4, 1, 1), (4, 2, 8), (4, 3, 2)]

n = 5

# dijkstra:

class Node:
	def __init__(self, vertex, weight=0):
		self.vertex = vertex
		self.weight = weight

	def __lt__(self, other):
		return self.weight < other.weight


class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest, weight) in edges:
			self.adj_list[src].append((dest, weight))


def get_route(prev, i, route):
	if i >= 0:
		get_route(prev, prev[i], route)
		route.append(i)

import sys #<--- if you use -1 instead, the relaxation function might have trouble
import heapq
def dijkstra(graph, source, n):
	pq = []
	heapq.heappush(pq, Node(source))

	visited = [False] * n
	visited[source] = True

	dist = [sys.maxsize] * n
	dist[source] = 0

	prev = [-1] * n

	route = []

	while pq:
		node = heapq.heappop(pq)
		u = node.vertex

		for (v, weight) in graph.adj_list[u]:
			if not visited[v] and dist[v] > dist[u] + weight:
				dist[v] = dist[u] + weight
				prev[v] = u
				heapq.heappush(pq, Node(v, weight))
		visited[u] = True


	for i in range(n):
		if i != source and dist[i] != sys.maxsize:
			get_route(prev, i, route)
			print(f'Path {source} --> {i}: minimum cost = {dist[i]}, route = {route}')
			route.clear()


graph = Graph(edges, n)

for source in range(n):
	dijkstra(graph, source, n)

# time: O(e * log v)
# space: O(v) (? it's the space required for the queue plus the additional data structures for the visited, distance, and route, if applicable. )




							# bellman ford algo:

# this is conceptually similar to dijkstra's algorithm in that it finds the shortest path from a node to all other nodes in a weighted graph, but unilke dijkstra's, bellman ford can deal with negative values as the weights. 

# it's a bit slower than dijkstra's for the same problem, but is more versatile due to the negative weight handling ability.

# if a graph contains a negative cycle (a cycle whose edges sum to a negative value) that is reachable from the source, then there is no shortest path; any path that has a point on the negative cycle can be made cheaper by one more trip through the negative cycle. in this case, bellman-ford can detect and report the negative cycle. 

# both dijkstra and bellman-ford use relaxation, where the approximation of the correct distance is replaced by a better one until they reach the solution. in both, the approximation is always an overestimate of the true distance, and is replaced by the minimum of its old dvalue and the length of a newly found path. 
	# however! dijkstra uses a priority queue to greedily select the closest unvisited node, and performs the relaxation step on all its outgoing edges. 
	# the bellman-ford, in contrast, simply relaxes all the edges. it does this v - 1 times (where v is number of vertices), because the longest possible path without a cycle can be v - 1 edges. 
		# in each of these iterations in bellman-ford, the number of vertices with correctly calculated distances grows, such that eventually all vertices will have their correct distances. the intermediate answers depend on the order of the edges being relaxed, but the final answer is the same. 
		# after the shortest path possible after scanning v - 1 times, a final scan is performed and if any distance is updated, then a path of length v edges has been found, which must mean that at least 1 negative weight cycle exists. 

# time: O(v * e)


# (u, v, w) represent edge from vertex `u` to vertex `v` having weight `w`
# edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
#         (4, 1, 1), (4, 2, 8), (4, 3, 2)]

# n = 5

# of graph edges as per the above diagram
edges = [
    # (x, y, w) —> edge from `x` to `y` having weight `w`
    (0, 1, -1), (0, 2, 4), (1, 2, 3), (1, 3, 2),
    (1, 4, 2), (3, 2, 5), (3, 1, 1), (4, 3, -3)
]

# set the maximum number of nodes in the graph
n = 5


def get_route(prev, vertex):
	if vertex >= 0:
		return get_route(prev, prev[vertex]) + [vertex]
	return []

import sys
def bellman_ford(edges, source, n):
	# step 1: initialize graph
	dist = [sys.maxsize] * n
	prev = [-1] * n
	dist[source] = 0

	# step 2: relax edges repeatedly
	for k in range(n - 1):
		for (u, v, weight) in edges:
			# if the distance to destination 'v' can be shortened by taking edge (u, v)
			if dist[u] != sys.maxsize and dist[v] > dist[u] + weight:
				dist[v] = dist[u] + weight
				prev[v] = u

	# step 3: run relaxation step once more for nth time to check for negative-weight cycles
	for (u, v, weight) in edges:
		if dist[u] != sys.maxsize and dist[v] > dist[u] + weight:
			print('graph contains a negative-weight cycle')
			return

	# step 4: for each vertex, print the distance between every vertex and every other vertex, as well as the path. 
	for i in range(n):
		if i != source and dist[i] < sys.maxsize:
			print(f'The distance of vertex {source} from vertex {i} is {dist[i]}. Its path is', get_route(prev, i))

	# return dist, prev

for source in range(n):
	bellman_ford(edges, source, n)

# time: O(v * e)




# of graph edges as per the above diagram
edges = [
    # (x, y, w) —> edge from `x` to `y` having weight `w`
    (0, 1, -1), (0, 2, 4), (1, 2, 3), (1, 3, 2),
    (1, 4, 2), (3, 2, 5), (3, 1, 1), (4, 3, -3)
]

# set the maximum number of nodes in the graph
n = 5



def get_route(prev, vertex):
	if vertex < 0:
		return []
	return get_route(prev, prev[vertex]) + [vertex]

import sys
def bellman_ford(edges, source, n):
	dist = [sys.maxsize] * n
	dist[source] = 0
	prev = [-1] * n

	for k in range(n - 1):
		for (u, v, weight) in edges:
			if dist[u] != sys.maxsize and dist[v] > dist[u] + weight:
				dist[v] = dist[u] + weight
				prev[v] = u

	for i in range(n):
		if dist[u] != sys.maxsize and dist[v] > dist[u] + weight:
			print('negative cycle detected.')
			return

	for i in range(n):
		if i != source and dist[i] < sys.maxsize:
			print(f'the distance from {source} to {i} is {dist[i]}. The path is {get_route(prev, i)}')

for source in range(n):
	bellman_ford(edges, source, n)






# bellman ford practice
#_______________________________________________________________


print('\n')
# of graph edges as per the above diagram
edges = [
    # (x, y, w) —> edge from `x` to `y` having weight `w`
    (0, 1, -1), (0, 2, 4), (1, 2, 3), (1, 3, 2),
    (1, 4, 2), (3, 2, 5), (3, 1, 1), (4, 3, -3)
]

# set the maximum number of nodes in the graph
n = 5

def get_route(prev, vertex):
	if vertex < 0:
		return []
	return get_route(prev, prev[vertex]) + [vertex]

import sys
def bellman_ford(edges, source, n):
	dist = [sys.maxsize] * n
	dist[source] = 0
	prev = [-1] * n

	for k in range(n - 1):
		for (u, v, weight) in edges:
			if dist[u] != sys.maxsize and dist[v] > dist[u] + weight:
				dist[v] = dist[u] + weight
				prev[v] = u

	for (u, v, weight) in edges:
		if dist[u] != sys.maxsize and dist[v] > dist[u] + weight:
			print('at least one negative cycle detected.')
			return

	for i in range(n):
		if i != source and dist[i] < sys.maxsize:
			print(f'the distance from vertex {source} to vertex {i} is {dist[i]}. the path is {get_route(prev, i)}')

for source in range(n):
	bellman_ford(edges, source, n)







# topo sort revisitation
edges = [(0, 6), (1, 2), (1, 4), (1, 6), (3, 0), (3, 4), (5, 1), (7, 0), (7, 1)]
n = 8

# edges = [
#     # (x, y, w) —> edge from `x` to `y` having weight `w`
#     (0, 1, -1), (0, 2, 4), (1, 2, 3), (1, 3, 2),
#     (1, 4, 2), (3, 2, 5), (3, 1, 1), (4, 3, -3)
# ]
# n = 5

class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		self.in_degree = [0] * n

		for (src, dest) in edges:
			self.adj_list[src].append((dest))
			self.in_degree[dest] += 1

graph = Graph(edges, n)


from collections import deque
def topological_sort(graph, n):
	topo_sort = []
	no_incoming = deque([i for i in range(n) if graph.in_degree[i] == 0])

	while no_incoming:
		vertex = no_incoming.popleft()
		topo_sort.append(vertex)

		for m in graph.adj_list[vertex]:
			graph.in_degree[m] -= 1
			if graph.in_degree[m] == 0:
				no_incoming.append(m)
	# print(graph.in_degree)

	for i in range(n):
		if graph.in_degree[i]:
			print('at least one negative cycle detected. topological sort not possible.')
			return None

	return topo_sort

result = topological_sort(graph, n)

if result:
	print('one valid topo sort result:', result)

# time: O(v + e)
# space: O(v)




# bellman ford practice
#_______________________________________________________________








						# A* Algorithm

# A* is a graph traversal and path search algorithm. 
# it's known for its completeness, optimality, and efficiency.
# it is the advanced form of the BFS algo.
# it is complete: 
	# it will find all available paths from start to end
# it is optimal:
	# it will find the least cost from the start to the end
	# (bfs is optimal if all costs are equal. if weighted, bfs is not optimal, it will simply return a valid path.)
# time complexity is dependent on the heuristic used. 
# a major drawback is its O(b^d) (also known as O(v)) space complexity, because it stores all nodes in memory. 
	# in practical travel-routing systems, this causes it to be outperformed by algos which can pre-process the graph to attain better performaance, as well as memory-bounded approaches. 
	# even so, A* is often still the best solution. 

# it is an informed search algorithm, or a best-first search. 
	# this means it applies to weighted graphs.
	# starting from a specific starting node, it aims to find a path to the given goal node having the smallest cost (least distance traveled, shortest time, etc).
	# informed search means the algo has extra information to begin with. 
		# ex: uninformed would be walking home blind. 
		# informed would be using your sight to see what path will bring you closer, or a map to know how far each point is from your destination.
		# A* will only perform a step if it seems promising and reasonable, unlike dijkstra, bfs, dfs.
	# it does this by maintaining a tree of paths originating at the start node and extending those paths on one edge at a time until its termination criterion is satisfied. 
	# at each iteration of its main loop, A* needs to determine which of its paths to extend. it does this based on the cost of the path and an estimate of the cost required to extend the path all the waya to the goal.
		# it selects the path that minimizes: f(n) = g(n) + h(n) where g(n) is the distance from start node to current node, and h(n) is a heuristic function that estimates the cost of the cheapest path from current node n to the goal.
		# it terminates when the path it chooses to extend is a path from start to goal or if there are no paths eligible to be extended.
		# the heuristic function is problem-specific.
		# if the heuristic function is admissible (it never overestimates the actual cost to get to the goal) then A* is guaranteed to return a least-cost path from start to goal.

# typical implementations use a priority queue to find the minimum cost nodes to expand. 
# at each step, the node with the lowest f(x) value is removed from the queue and f and g values are updated accordingly. these neighbors are then added to the queue. 
# the algo continues until a removed node (by definition the one with the lowest cost) is a goal node. 

#______________________________________________________________________
# a note on A* vs BFS:
	# bfs uses a queue while A* uses a priority queue. in general, queues are much faster than priority queues (remove is O(1) in queue vs O(log n) in priority queue). A* normally expands far fewer nodes than bfs, so A* will generally be faster. if this isn't the case, then bfs will be faster. 
#______________________________________________________________________

# f(n): the actual cost path from the start node to the goal node.
# g(n): the actual cost path from the start node to the current node.
# h(n): the actual cost path from the current node to the goal node.


print('\n')

from collections import deque

# edges = [('A')]

# class Graph:
# 	def __init__(self, edges, n):
# 		self.adj_list = [[] for _ in range(n)]:
# 		for (src, dest, weight) in edges:
# 			self.adj_list[src].append((dest, weight))

adjacency_list = {
    'A': [('B', 1), ('C', 3), ('D', 7)],
    'B': [('D', 5)],
    'C': [('D', 12)]
    }

class Graph:
	def __init__(self, adj_list):
		self.adj_list = adj_list

	def get_neighbors(self, v):
		return self.adj_list[v]

	# heuristic function with equal values for all nodes
	def h(self, n):
		H = {
			'A': 1,
			'B': 1,
			'C': 1,
			'D': 1
		}
		return H[n]

	# def h_manhattan(self, n):


	def a_star(self, start_node, stop_node):
		# open list is a list of nodes which have been visited, but whose neighbors haven't been. 
		# begins with the start node.
		# closed list is a list of nodes which have been visited and whose neighbors also have. 
		open_list = set([start_node])
		closed_list = set([])

		# g contains the current distances from start_node to all other nodes
		# the default valule if not found in the map is infinity
		g = {}
		g[start_node] = 0

		# parents contains an adjacency map of all nodes
		parents = {}
		parents[start_node] = start_node

		while len(open_list) > 0:
			n = None
			# find a node with the lowest value of f() - evaluation function
			for v in open_list:
				# if n == None or g[v] + self.h(v) < g[n] + self.h(n):
				# 	n = v
				if n == None or g[v] + self.h_manhattan(v) < g[n] + self.h_manhattan(n):
					n = v

			if n == None:
				print('Path does not exist')
				return None

			# if the current node is the stop_node:
			# we begin reconstructing the path from it to the start node.
			if n == stop_node:
				route = []
				while parents[n] != n:
					route.append(n)
					n = parents[n]
				route.append(start_node)
				route.reverse()
				print(f'Path found: {route}')
				return route

			# for all neighbors of the current node:
			for (m, weight) in self.get_neighbors(n):
				# if current node isn't in both open and closed lists:
				# add it to the open list and note n as its parent 
				if m not in open_list and m not in closed_list:
					open_list.add(m)
					parents[m] = n
					g[m] = g[n] = weight

				# otherwise, see if it's quicker to first visit n, then m
				# if it is, update parent data and g data
				# if the node was in the closed_list, move it to open_list.
				else:
					if g[m] > g[n] + weight:
						g[m] = g[n] + weight
						parents[m] = n

						if m in closed_list: 
							closed_list.remove(m)
							open_list.add(m)

			# remove n from the open list and add it to the closed list
				# because all n's neighbors have been visited
			open_list.remove(n)
			closed_list.add(n)

		print('Path doesn\'t exist.')
		return None

# graph = Graph(adjacency_list)
# graph.a_star('A', 'D')

# noteworthy:
# if using a grid where only up, down, left, and right are allowed moves (the Von Neumann neighborhood), a good heuristic to use is the Manhattan distance. this is also known as city block distance, or rectilinear distance. distance = |x1 - x2| + |y1 - y2|.
# if the grid allows for diagonal moves as well, a good heuristic to use is the Euclidean distance. 
# distance = sqroot((x2 - x1) ** 2 + (y2 - y1) ** 2)







print('\n')

#______________________________________________________________________
#______________________________________________________________________
#______________________________________________________________________

# personal challenge: find shortest path given a maze, start, and end points.
# extra practice 
maze = [
	[0, 1, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0],
	[0, 1, 0, 1, 0, 0],
	[0, 1, 0, 0, 1, 0],
	[0, 0, 0, 0, 1, 0]
]

start = [0, 0]
# end = [4, 5]
end = [0, 5]
end = [3, 3]


def get_route(prev, i, route):
	if i[0] >= 0 and i[1] >= 0:
		get_route(prev, prev[i[0]][i[1]], route)
		route.append(i)


from collections import deque
def bfs_path(maze, start, end):

	row, col = len(maze), len(maze[0])

	queue = deque([(start[0], start[1])])

	visited = set([(start[0], start[1])])

	dist = [[-1 for col in range(col)] for row in range(row)]
	dist[start[0]][start[1]] = 0

	prev = [[(-1, -1) for col in range(col)] for row in range(row)]
	route = []

	while queue:
		node = queue.popleft()
		for neighbor in [
			(node[0] + 1, node[1]),
			(node[0], node[1] + 1),
			(node[0] - 1, node[1]),
			(node[0], node[1] - 1)
		]:
			if neighbor not in visited and 0 <= neighbor[0] < row and 0 <= neighbor[1] < col:
				if maze[neighbor[0]][neighbor[1]] == 1:
					continue

				visited.add((neighbor[0], neighbor[1]))
				dist[neighbor[0]][neighbor[1]] = 1 + dist[node[0]][node[1]]
				prev[neighbor[0]][neighbor[1]] = (node[0], node[1])
				queue.append((neighbor[0], neighbor[1]))

				if [neighbor[0], neighbor[1]] == end:
					get_route(prev, neighbor, route)
					return f'The distance from {start} to {end} is {dist[end[0]][end[1]]}. The route is {route}.'


print('single shortest bfs path in grid:', bfs_path(maze, start, end))






# same problem, but with descriptions of the logic 
maze = [
	[0, 1, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0],
	[0, 1, 0, 1, 0, 0],
	[0, 1, 0, 0, 1, 0],
	[0, 0, 0, 0, 1, 0]
]



start = [0, 0]
end = [4, 5]
end = [0, 5]
# end = [3, 3]


def get_route(prev, i, route):
	# if row >= 0 and col >= 0:
	if i[0] >= 0 and i[1] >= 0:
		get_route(prev, prev[i[0]][i[1]], route)
		route.append(i)


from collections import deque
def bfs(maze, start, end):

	# 1. create a queue to guide bfs traversal
	queue = deque([(start[0], start[1])])

	# 2. create structures to keep track of visited vertices, distances and route
	row, col = len(maze), len(maze[0])
	# print(row, col)

	# can use a set or an array, initialize with starting point
	visited = set([(start[0], start[1])])

	# visited = [[False for i in range(col)] for j in range(row)]
	# print('visited', visited)
	# visited[0][0] = True
	# print(visited)

	dist = [[-1 for col in range(col)] for row in range(row)]
	# initialize the starting distance to 0
	dist[start[0]][start[1]] = 0

	# prev = [[-1 for row in range(row)] for col in range(col)]
	prev = [[(-1, -1) for col in range(col)] for row in range(row)] #<--not sure how to structure route yet

	route = []

	# 3. remove from queue the starting node and identify neighbors. if it is a valid position in the map and not visited and not a wall, increase the neighbor's distance by 1 and add that neighbor to the queue, add to visited set, and to prev. 

	while queue:
		node = queue.popleft()
		for neighbor in [
			(node[0] + 1, node[1]),
			(node[0], node[1] + 1),
			(node[0] - 1, node[1]),
			(node[0], node[1] - 1)
		]:
			# if the neighbor is not in visited and is within the bounds of the matrix:
			if neighbor not in visited and 0 <= neighbor[0] < row and 0 <= neighbor[1] < col:
					# if the neighbor is a wall, disregard this neighbor
					if maze[neighbor[0]][neighbor[1]] == 1:
						continue
					# update the neighbor's distance, add to visited, add to queue, update neighbor's previous node to be the current node.
					dist[neighbor[0]][neighbor[1]] = dist[node[0]][node[1]] + 1
					visited.add((neighbor[0], neighbor[1]))
					queue.append((neighbor[0], neighbor[1]))
					prev[neighbor[0]][neighbor[1]] = (node[0], node[1])
					# if the neighbor is the end point, get the route and return relevant values.
					if neighbor == (end[0], end[1]):
						get_route(prev, neighbor, route)
						# return f'the distance from {start} to {end} is {dist[neighbor[0]][neighbor[1]]}.'
						return f'The distance from {start} to {end} is {dist[neighbor[0]][neighbor[1]]}. The route is {route}.'
						
			visited.add((node[0], node[1]))
	# print(ret_val)
	# print('post prev', prev)
	# print('post dist', dist)

	### my attempt to get the route from every point to every other point: ###
	# route = []
	# for col in range(col):
	# 	for row in range(row):
	# 		print('rowwin, colllin', [row, col], 'distin', dist[row][col])
	# 		if [row, col] != start and dist[row][col] != -1:
	# 		# get_route(prev, row, col, route)
	# 			print('row', row, 'col', col)
	# 			print('mazin', maze[row][col])
	# 			print('prev row col', prev[row][col])
	# 			get_route(prev, prev[row][col], route)
	# 			# route.clear() 
	# print(f'the distance from {start} to {end} is {dist[end[0]][end[1]]}. the route is {route}')
	### my attempt to get the route from every point to every other point: ###

print(bfs(maze, start, end))



edges = [
        (1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (5, 9),
        (5, 10), (4, 7), (4, 8), (7, 11), (7, 12)
    ]
 
# total number of nodes in the graph (labelled from 0 to 14)
n = 13

class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest) in edges:
			self.adj_list[src].append(dest)
			self.adj_list[dest].append(src)


graph = Graph(edges, n)
visited = [False] * n
output = []


from collections import deque
def bfs_graph(graph, v, visited):
	queue = deque([v])
	while queue:
		node = queue.popleft()
		output.append(node)
		for neighbor in graph.adj_list[node]:
			if not visited[neighbor]:
				queue.append(neighbor)
		visited[node] = True
		
	print(output)


for v in range(n):
	if not visited[v]:
		bfs_graph(graph, v, visited)

print('\n')


row, col = len(maze), len(maze[0])

visited = set([(start[0], start[1])])

dist = [[-1 for col in range(col)] for row in range(row)]
dist[start[0]][start[1]] = 0

prev = [[(-1, -1) for col in range(col)] for row in range(row)]
route = []


def rec_dfs(maze, node, visited, dist, prev, route, end):

	if [node[0], node[1]] == end:
		get_route(prev, node, route)
		print(f'The distance from {start} to {end} is {dist[end[0]][end[1]]}. The route is {route}.')
		return True
		# return 'yes'
	
	for neighbor in [
		(node[0] + 1, node[1]),
		(node[0], node[1] + 1),
		(node[0] - 1, node[1]),
		(node[0], node[1] - 1)
	]:	
		if neighbor not in visited and 0 <= neighbor[0] < row and 0 <= neighbor[1] < col:
				if maze[neighbor[0]][neighbor[1]] != 1:
					visited.add((neighbor[0], neighbor[1]))
					dist[neighbor[0]][neighbor[1]] = 1 + dist[node[0]][node[1]]
					prev[neighbor[0]][neighbor[1]] = (node[0], node[1])
					if rec_dfs(maze, neighbor, visited, dist, prev, route, end)== True:
						return True
						# why does adding a value here along with True cause it to explore additional nodes instead of terminating early? 
						# because then the recursive function call's return value no longer satisfies that condition (rec_dfs no longer == True, it == True + y.) 
							# therefore, after the base case is met, other recursive calls still get to run because for all calls in which the base case is satisfied, the call returns a value that isn't the condition, so it disregards the if statement and moves on to the next logic, which is a return statement with the value False. 
				
	# print(len(visited))
	return False

print(rec_dfs(maze, start, visited, dist, prev, route, end))
# rec_dfs(maze, start, visited, dist, prev, route, end)
# calling the function evaluates to True, which allows the print statement in the function to execute, which is the information we really want. 

# make sure that you return from the for loop of neighbors once you've found the node you're looking for (this is provided in the if node == end condition at the top of the function).


print('\n')





def dfs_path(maze, start, end):

	row, col = len(maze), len(maze[0])

	visited = set([(start[0], start[1])])

	dist = [[-1 for col in range(col)] for row in range(row)]
	dist[start[0]][start[1]] = 0

	prev = [[(-1, -1) for col in range(col)] for row in range(row)]
	route = []

	def rec_dfs(maze, node, neighbor, visited, dist, prev, route, end):
		print(visited)
		print(neighbor)
		if neighbor not in visited and 0 <= neighbor[0] < row and 0 <= neighbor[1] < col:
			if maze[neighbor[0]][neighbor[1]] != 1:
				rec_dfs(maze, node, (node[0] + 1, node[1]), visited, dist, prev, route, end),
				rec_dfs(maze, node, (node[0], node[1] + 1), visited, dist, prev, route, end),
				rec_dfs(maze, node, (node[0] - 1, node[1]), visited, dist, prev, route, end),
				rec_dfs(maze, node, (node[0], node[1] - 1), visited, dist, prev, route, end)

		visited.add((neighbor[0], neighbor[1]))
		dist[neighbor[0]][neighbor[1]] = 1 + dist[node[0]][node[1]]
		prev[neighbor[0]][neighbor[1]] = (node[0], node[1])

		if [neighbor[0], neighbor[1]] == end:
			get_route(prev, neighbor, route)
			return f'The distance from {start} to {end} is {dist[end[0]][end[1]]}. The route is {route}.'

	rec_dfs(maze, start, start, visited, dist, prev, route, end)


# print(dfs_path(maze, start, end))


# graph = Graph(edges, n)
# visited = [False] * n
# output = []
# def rec_dfs(graph, v, visited):
# 	visited[v] = True
# 	output.append(v)

# 	for u in graph.adj_list[v]:
# 		if not visited[u]:
# 			rec_dfs(graph, u, visited)


		

	# while queue:
	# 	node = queue.popleft()
	# 	for neighbor in [
	# 		(node[0] + 1, node[1]),
	# 		(node[0], node[1] + 1),
	# 		(node[0] - 1, node[1]),
	# 		(node[0], node[1] - 1)
	# 	]:
	# 		if neighbor not in visited and 0 <= neighbor[0] < row and 0 <= neighbor[1] < col:
	# 			if maze[neighbor[0]][neighbor[1]] == 1:
	# 				continue

	# 			visited.add((neighbor[0], neighbor[1]))
	# 			dist[neighbor[0]][neighbor[1]] = 1 + dist[node[0]][node[1]]
	# 			prev[neighbor[0]][neighbor[1]] = (node[0], node[1])
	# 			queue.append((neighbor[0], neighbor[1]))

	# 			if [neighbor[0], neighbor[1]] == end:
	# 				get_route(prev, neighbor, route)
	# 				return f'The distance from {start} to {end} is {dist[end[0]][end[1]]}. The route is {route}.'

edges = [
        # Notice that node 0 is unconnected
        (1, 2), (1, 7), (1, 8), (2, 3), (2, 6), (3, 4),
        (3, 5), (8, 9), (8, 12), (9, 10), (9, 11)
    ]
 
# total number of nodes in the graph (labelled from 0 to 12)
n = 13

class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest) in edges:
			self.adj_list[src].append(dest)
			self.adj_list[dest].append(src)


visited = [False] * n
output = []
graph = Graph(edges, n)

def dfs(graph, node, visited):
	# a cool way to only traverse a certain number: make the termination condition whatever you want

	# if the base condition is satisfied, return the value "whaha"
	if len(output) == 5:
		# print('whaha')
		return 'whaha'
		# return True

	visited[node] = True
	output.append(node)

	# for each neighbor:
	for u in graph.adj_list[node]:
		if not visited[u]:
			# if the neighbor satsifies the base condition (the return value for the neighbor's recursive call is the same as the base condition's return value), return a new value, 'yeayyaa'. 
			if dfs(graph, u, visited) == 'whaha':
				return f'this return output is yeayyaa'

	return False
# for i in range(n):
# 	if not visited[i]:
# 		dfs(graph, i, visited)


print(dfs(graph, 1, visited))
print('yeah', output)





#________________________________________________________________#
#________________________________________________________________#
#________________________________________________________________#


















# Trying to implement dijkstra without using the node class. having trouble.
# (u, v, w) represent edge from vertex `u` to vertex `v` having weight `w`
edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
        (4, 1, 1), (4, 2, 8), (4, 3, 2)]

n = 5

# dijkstra:

class Node:
	def __init__(self, vertex, weight=0):
		self.vertex = vertex
		self.weight = weight

	def __lt__(self, other):
		return self.weight < other.weight


class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest, weight) in edges:
			self.adj_list[src].append((dest, weight))


def get_route(prev, i, route):
	if i >= 0:
		get_route(prev, prev[i], route)
		route.append(i)

import sys #<--- if you use -1 instead, the relaxation function might have trouble
import heapq
def dijkstra(graph, source, n):
	pq = []
	heapq.heappush(pq, (source, 0))

	visited = [False] * n
	visited[source] = True

	dist = [sys.maxsize] * n
	dist[source] = 0

	prev = [-1] * n

	route = []

	while pq:
		node = heapq.heappop(pq)
		u = node[0]
		# print('graph adj list u', graph.adj_list[u])
		for (v, weight) in graph.adj_list[u]:
			# print((v, weight))
			if not visited[v] and dist[v] > dist[u] + weight:
				dist[v] = dist[u] + weight
				prev[v] = u
				heapq.heappush(pq, (v, weight))
		visited[u] = True

	for i in range(n):
		# print(i)
		if i != source and dist[i] != sys.maxsize:
			get_route(prev, i, route)
			print(f'Path {source} --> {i}: minimum cost = {dist[i]}, route = {route}')
			route.clear()


graph = Graph(edges, n)

# for source in range(n):
# 	dijkstra(graph, source, n)



# understand adjacency matrix representation vs adj_list
# this might help: https://www.geeksforgeeks.org/python-program-for-dijkstras-shortest-path-algorithm-greedy-algo-7/







# more practice 

edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
        (4, 1, 1), (4, 2, 8), (4, 3, 2)]

n = 5

# dijkstra:

class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest, weight) in edges:
			self.adj_list[src].append((dest, weight))


class Node:
	def __init__(self, vertex, weight=0):
		self.vertex = vertex
		self.weight = weight

	def __lt__(self, other):
		return self.weight < other.weight


def get_route(prev, i, route):
	if i >= 0:
		get_route(prev, prev[i], route)
		route.append(i)


import heapq
import sys

def dijkstra(graph, source, n):
	queue = []
	heapq.heappush(queue, Node(source))

	visited = [False] * n
	visited[source] = True

	dist = [sys.maxsize] * n
	dist[source] = 0

	prev = [-1] * n

	while queue:
		node = heapq.heappop(queue)
		u = node.vertex

		for (v, weight) in graph.adj_list[u]:
			if not visited[v] and dist[v] > dist[u] + weight:
				dist[v] = dist[u] + weight
				prev[v] = u
				heapq.heappush(queue, Node(v, weight))
		visited[u] = True

	route = []
	for i in range(n):
		if i != source and dist[i] < sys.maxsize:
			get_route(prev, i, route)
			print(f'Path {source} --> {i}: minimum cost = {dist[i]}. The route is {route}.')
			route.clear()

# for source in range(n):
# 	dijkstra(graph, source, n)



