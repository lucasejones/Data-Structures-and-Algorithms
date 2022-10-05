
# greedy algorithms

# a greedy algorithm is one that chooses local optimums to reach a solution instead of exhaustively searching for the global optimum. this reduces computational work in exchange for a slightly less than optimal solution. However, this tradeoff is generally acceptable because a greedy algorithm arrives functionally close enough to the optimal, and the computational savings are significant.

# helpful article: https://medium.com/techie-delight/top-7-greedy-algorithm-problems-3885feaf9430

# common problems are:
	# the activity selection problem
	# the graph coloring problem
	# the job sequencing problem with deadlines
	# find minimum platforms needed to avoid delay in the train arrival
	# huffman coding compression algorithm 
	# dijkstra's algorithm (single-source shortest paths) (think this is solution to bridges of konningsburg?)
	# kruskal's algorithm for finding minimum spanning tree



#####_____________________________________________________________________#####
#####_____________________________________________________________________#####
#####_____________________________________________________________________#####

					# activity selection problem

# Given a set of activities, along with the starting and finishing time of each activity, find the maximum number of activities performed by a single person assuming that a person can only work on a single activity at a time. 

activities = [
	(3, 8), (5, 9), (6, 10), (8, 11), (8, 12), (2, 13), 
	(12, 14), (1, 4), (3, 5), (0, 6), (5, 7)
	]

# concise version with no quicksort
def greedy(activities):
	activities.sort(key=lambda x: x[1])
	S = set([activities[0]])
	k = 0

	for i in range(1, len(activities)):
		if activities[i][0] >= activities[k][1]:
			S.add(activities[i])
			k = i

	return S

result = list(greedy(activities))
result.sort(key=lambda x: x[1])
print(result)

# time: O(n log n), where n is number of activities.
# space: O(1)

# the thinking is:
	# instead of doing a dfs search to find the optimal path, you can instead use a greedy algorithm. 
	# first, sort by ascending finish times because this gives you the ability to add the earliest starting activites.  
	# then create a set that will hold all the selected activities. 
	# then compare each start time to the last selected finishing time and as soon as you see one, add it to the set of selected activities.



# if you want to perform a weighted activity selection problem, dynamic programming will get the job done.
# this is a good reference: https://www.cs.princeton.edu/~wayne/cs423/lectures/dynamic-programming-4up.pdf





#####_____________________________________________________________________#####
#####_____________________________________________________________________#####
#####_____________________________________________________________________#####

					# the graph coloring problem


# this is a helpful resource for this problem: https://www.techiedelight.com/greedy-coloring-graph/

# graph coloring is a way to identify a graph's vertices such that no two adjacent vertices share the same label.
# we can do this using a greedy algorithm and minimize the number of colors used. 

# terminology:
#_____________
# k-colorable graph: 
	# a coloring using at most k colors is called a proper k-coloring, and a graph that can be assigned a proper k-coloring is k-colorable.

# k-chromatic graph: 
	# the smallest number of colors neeed to color a graph G is called its chromatic number.
	# a graph is k-chromatic if its chromatic number is exactly k. 

# Brooks' theorem:
	# states the relationship between the maximum degree of a graph and its chromatic number.
	# a connected graph can be colored with only x colors, where every vertex has at most x neighbors, except for complete graphs and graphs containing an odd length cycle, which requires x+1 colors. 

# greedy coloring considers the vertices of the graph in sequence and assigns each vertex its first available color.

# greedy coloring doesn't always use the minimum number of colors possible to color a graph.
	# for a graph of max degrees x, greedy coloring will use at most x + 1 colors.
	# a complete bipartite graph can require only 2 colors, while a greedy algo will instead color n/2 colors (which in the example given is 4 colors; 1 color more than the # of most edges, which is 3 in the example).



# building the undirected graph object:

class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]

		for (src, dest) in edges:
			self.adj_list[src].append(dest)
			# this line applies for UNdirected graphs. 
			# unlike this application, topo sort uses directed graphs, so it doesn't care about where the nodes come from, just where they're going.
			self.adj_list[dest].append(src)


 # List of graph edges
edges = [(0, 1), (0, 4), (0, 5), (4, 5), (1, 4), (1, 3), (2, 3), (2, 4)]

# total number of nodes in the graph (labelled from 0 to 5)
n = 6

graph = Graph(edges, n)
print(graph.adj_list)

colors = ['blue', 'green', 'red', 'yellow', 'black', 'white']

def color_graph(graph, n):
	# keep track of the color assigned to each vertex
	result = {}

	# assign a color to vertex one by one
	for u in range(n):

		# check colors of adjacent vertices of 'u' and store them in a set
		assigned = set([result.get(i) for i in graph.adj_list[u] if i in result])

		# print('u, assigned', u, graph.adj_list[u], assigned)

		# check for the first free color
		color = 0

		for c in assigned:

			if color != c:
				break
			color += 1

		# assign vertex 'u' the first available color
		result[u] = color

	for v in range(n):
		print(f'the color assigned to vertex {v} is {colors[result[v]]}')

	return result

print(color_graph(graph, n))


# time: O(v * e) where v and e are the number of vertices and edges.
# space: ? (less than max neighbors for a vertex)

# applications: pattern matching, designing seating plans, scheduling exam timetable, solving sudokus, etc

print('\n' * 6)
print('________')
print('\n' * 4)



#####_____________________________________________________________________#####
#####_____________________________________________________________________#####
#####_____________________________________________________________________#####

			# Dijkstra's Algorithm (single-source shortest paths)


# originally, this algorithm was developed to find the shortest path between any two nodes. 
# it can more commonly be used to take any given node as the source, and find the shortest path to any other node. this produces a minimum-spanning tree. 
# it can stop running once the distance to a specific node is determined as well. 

# it uses labels that are positive ints or real numbers, which are totally ordered (all comparable). it can be generalized to use any labels that are partially ordered, so long as the subsequent labels are non-decreasing. this is called the dijkstra shortest-path algorithm. 

# the idea is to store distances as you go and query partial solutions, updating with new distance information. this prevents the need to recalculate, and is a hallmark of dynamic programming (called memoization). dijkstra's accomplishes this with a min-priority queue (though you can also do it with an array, though this takes O(n^2 time)). 
# more accurate values gradually replace an approximation to the correct distance until the shortest distance is reached.

# the time complexity is O((v + e) log v) where v is number of nodes and e is edges. 

# a variant of this is known as uniform cost search, and is useful in AI.

# this link points to the source educational material:
# https://www.techiedelight.com/single-source-shortest-paths-dijkstras-algorithm/

# (u, v, w) represent edge from vertex `u` to vertex `v` having weight `w`
edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
        (4, 1, 1), (4, 2, 8), (4, 3, 2)]

# total number of nodes in the graph (labelled from 0 to 4)
n = 5



import sys
import heapq 


# class to store a heap node
class Node:
	def __init__(self, vertex, weight=0):
		self.vertex = vertex
		self.weight = weight

	def __lt__(self, other):
		return self.weight < other.weight

# creating the graph from provided edges and vertex number
class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest, weight) in edges:
			self.adj_list[src].append((dest, weight))

# will recursively append predecessor nodes to a route list
def get_route(prev, i, route):
	if i >= 0:
		get_route(prev, prev[i], route)
		route.append(i)

def find_shortest_paths(graph, source, n):
	# create a min-heap and push source node having distance 0
	pq = []
	heapq.heappush(pq, Node(source))

	# set initial distance from the source to 'v' as infinity
	dist = [sys.maxsize] * n

	# distance from the source itself is 0
	dist[source] = 0

	# list to track vertices for which minimum cost is already found
	done = [False] * n
	done[source] = True

	# stores predecessor of a vertex (to a print path)
	prev = [-1] * n

	# run until min-heap is empty
	while pq:
		node = heapq.heappop(pq) 	# remove and return the best vertex
		u = node.vertex				# get the vertex number

		# do for each neighbor 'v' of 'u'
		for (v, weight) in graph.adj_list[u]:
			# if the neighbor is unvisited and the distance through the current vertex is less than the distance of the neighbor:
			if not done[v] and (dist[u] + weight) < dist[v]:
				# the new distance for the neighbor is the current vertex's distance from source + the distance to the neighbor
				dist[v] = dist[u] + weight
				# the predecessor of the neighbor is the current vertex
				prev[v] = u
				# push into the heap the neighbor and its updated distance
				heapq.heappush(pq, Node(v, dist[v]))

							## this works as well:
			# if not done[v]:
			# 	dist[v] = min(dist[u] + weight, dist[v])
			# 	prev[v] = u
			# 	heapq.heappush(pq, Node(v, dist[v]))

		# once all neighbors are visited, mark the current vertex as done so it won't get picked up again
		done[u] = True

	# create a list to hold the route for the minimum distance
	route = []
	for i in range(n):
		# if the vertex isn't the source and the distance has been updated at least once:
		if i != source and dist[i] != sys.maxsize:
			# run the get route function to populate the route list
			get_route(prev, i, route)
			print(f'Path {source} -> {i}: minimum cost = {dist[i]}, route = {route}')
			# reset the route list to empty for the next vertex
			route.clear()


graph = Graph(edges, n)

for source in range(n):
	find_shortest_paths(graph, source, n)

# time: O(e * log v)
# refer to this: https://stackoverflow.com/questions/26547816/understanding-time-complexity-calculation-for-dijkstra-algorithm












#####_____________________________________________________________________#####
#####_____________________________________________________________________#####
#####_____________________________________________________________________#####

								# practice!


print('\n' * 10)
print('_______')
print('\n' * 4)

# solve activity selection problem 
activities = [
	(3, 8), (5, 9), (6, 10), (8, 11), (8, 12), (2, 13), 
	(12, 14), (1, 4), (3, 5), (0, 6), (5, 7)
	]


def greedy_activity(activities):
	# sort by finishing times
	activities.sort(key=lambda x: x[1])

	# create set of selected times, initialized with the first-finishing time
	selected = set([activities[0]])
	# set k to 1
	k = 1

	# compare remaining activities to find soonest beginning and add to selected
	for i in range(1, len(activities)):
		if activities[i][0] >= activities[k][1]:
			selected.add(activities[i])
			k = i

	return selected

result = list(greedy_activity(activities))
result.sort(key=lambda x: x[1])
# print(result)

# time: O(n log n)
# space: ? O(n / 2)








# solve graph coloring problem


edges = [(0, 1), (0, 4), (0, 5), (4, 5), (1, 4), (1, 3), (2, 3), (2, 4)]
n = 6


# the goal is to color each vertex such that no two adjacent vertices are the same color, and to approach the minimal number of colors necessary to complete this task. greedy algorithm can do this approachment, and at worst will use v+1 colors, where v is the max number of degrees for a given vertex.

# creating the directed graph
class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest) in edges:
			self.adj_list[src].append(dest)
			self.adj_list[dest].append(src)

graph = Graph(edges, n)
# print('adj list', graph.adj_list)

# colors list to assign each vertex
colors = ['blue', 'green', 'red', 'black', 'white']


def coloring_problem(graph, edges, n):
	# dict of colors assigned to a given set of neighbors
	result = {}

	# for each vertex:
	for u in range(n):
		# create a set of each vertex's neighbors, where should they have a color assigned to them, the set contains that color
		assigned = set([result[i] for i in graph.adj_list[u] if i in result])

		# assigning the first available color
		color = 0
		for c in assigned:
			if color != c:
				break
			color += 1
		result[u] = color

	for v in range(n):
		print(f'the color assigned to {v} is {colors[result[v]]}')

	# return result

# print(coloring_problem(graph, edges, n))

# time: O(v * e)
# space: O(1)









# perform dijkstra's algorithm for all vertices based on their weights 

# (u, v, w) represent edge from vertex `u` to vertex `v` having weight `w`
edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
        (4, 1, 1), (4, 2, 8), (4, 3, 2)]

# total number of nodes in the graph (labelled from 0 to 4)
n = 5

# make a node class
# make a graph class
# create a get_route function
# make algorithm that outputs distance from a source to each vertex 


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


import sys
import heapq

def dijkstra(graph, source, n):
	pq = []
	heapq.heappush(pq, Node(source))

	dist = [sys.maxsize] * n
	dist[source] = 0

	visited = [False] * n
	visited[source] = True

	prev = [-1] * n

	while pq:
		node = heapq.heappop(pq)
		u = node.vertex

		for (v, weight) in graph.adj_list[u]:
			if not visited[v] and dist[v] > dist[u] + weight:
				dist[v] = dist[u] + weight
				prev[v] = u
				heapq.heappush(pq, Node(v, dist[v]))
		visited[u] = True

	route = []
	for i in range(n):
		if i != source and dist[i] != sys.maxsize:
			get_route(prev, i, route)
			print(f'Path {source} --> {i}: minimum cost = {dist[i]}, route = {route}')
			route.clear()


graph = Graph(edges, n)

for source in range(n):
	dijkstra(graph, source, n)


# why does dijkstra get the optimal solution every time? 
	# it goes through each node and all that node's neighbors
	# even though it does this greedily (going to shortest distance to any neighbor), it still exhausts the search space.
	# this provides the optimal solution, despite being greedy. they are not exclusive concepts.


print('\n' * 8)








# solve activity selection problem

activities = [
	(3, 8), (5, 9), (6, 10), (8, 11), (8, 12), (2, 13), 
	(12, 14), (1, 4), (3, 5), (0, 6), (5, 7)
	]


def greedy_activity_selection(activities):
	# sort by finishing times
	# initialize a set of selected activities with the first finishing one
	# iterate through to find the soonest beginning and add to selected
	# return selected
	activities.sort(key=lambda x: x[1])
	selected = set([activities[0]])
	k = 1
	for i in range(1, len(activities)):
		if activities[i][0] >= activities[k][1]:
			selected.add(activities[i])
			k = i
	return selected

result = list(greedy_activity_selection(activities))
result.sort(key=lambda x: x[1])
print(result)




# solve graph coloring problem


edges = [(0, 1), (0, 4), (0, 5), (4, 5), (1, 4), (1, 3), (2, 3), (2, 4)]
n = 6

# create undirected graph class
class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest) in edges:
			self.adj_list[src].append(dest)
			self.adj_list[dest].append(src)

colors = ['blue', 'green', 'red', 'black', 'white']

def coloring(graph, n):
	# color graph such that no adjacent vertices have the same color
	# dictionary to store the relationships between vertices and their colors
	result = {}

	# for each vertex
	for u in range(n):
		# if a neighbor is colored, create a set containing those colors
		assigned = set([result[i] for i in graph.adj_list[u] if i in result])

		# find the first available color and use it as value for the vertex key in result
		color = 0
		for c in assigned:
			if c != color:
				break
			color += 1
		result[u] = color

	for i in range(n):
		print(f'the color for vertex {i} is {colors[result[i]]}')


graph = Graph(edges, n)
print(coloring(graph, n))

# time: O(v * e)





# perform dijkstra's algorithm for all vertices based on their weights 

# (u, v, w) represent edge from vertex `u` to vertex `v` having weight `w`
edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
        (4, 1, 1), (4, 2, 8), (4, 3, 2)]

n = 5

# create node class
# create directed graph class
# create get route function
# create dijkstra's algorithm

# the idea is to identify a source and a destination, and determine the shortest path between them by greedily selecting the path from the source to its nearest neighbor. all distances are initialized with infinity. the distance of that neighbor is then updated with whatever is smaller: the distance it currently has or the distance through the current node. Eventually, the shortest path is found and the route is traced back through a route function.


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


import sys
import heapq

def dijkstra(graph, source, n):
	pq = []
	heapq.heappush(pq, Node(source))

	dist = [sys.maxsize] * n
	dist[source] = 0

	visited = [False] * n
	visited[source] = True

	prev = [-1] * n

	while pq:
		node = heapq.heappop(pq)
		u = node.vertex

		for (v, weight) in graph.adj_list[u]:
			# this is the relaxation condition
			if not visited[v] and dist[v] > dist[u] + weight:
				dist[v] = dist[u] + weight
				prev[v] = u
				heapq.heappush(pq, Node(v, weight))
		visited[u] = True

	route = []
	for i in range(n):
		if i != source and dist[i] != sys.maxsize:
			get_route(prev, i, route)
			print(f'path {source} --> {i}: minimum cost = {dist[i]}, route = {route}')
			route.clear()


graph = Graph(edges, n)

for source in range(n):
	dijkstra(graph, source, n)

# gives you optimal path to all destinations from single source instead of every source
# dijkstra(graph, 1, n)


	

print('\n' * 7)


# (u, v, w) represent edge from vertex `u` to vertex `v` having weight `w`
edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
        (4, 1, 1), (4, 2, 8), (4, 3, 2)]

n = 5


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



import sys
import heapq

def dijkstra(graph, source, n):
	pq = []
	heapq.heappush(pq, Node(source))

	dist = [sys.maxsize] * n
	dist[source] = 0

	visited = [False] * n
	visited[source] = True

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



		# for finding a specific path:
		# if u == target:
		# 	get_route(prev, u, route)
		# 	return print(f'path {source} --> {u}: min cost: {dist[u]}, route: {route}')

	for i in range(n):
		if i != source and dist[i] != sys.maxsize:
			get_route(prev, i, route)
			print(f'Path {source} --> {i}: minimum cost = {dist[i]}, route = {route}')
			route.clear()



graph = Graph(edges, n)

print('dijkstra')
for source in range(n):
	dijkstra(graph, source, n)

# for finding a specific path:
# dijkstra(graph, 1, 3, n)
# time: O(e log v)







print('\n' * 7)

# BFS works for unweighted, and either directed or undirected 
# dijkstra works for weighted (positive numbers only) and either directed or undirected
# bellman-ford works for weighted (negative numbers and positive numbers) and either directed or undirected

# BFS practice for comparison to dijkstra (bfs works for unweighted)
edges = [(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 2),
        (4, 1), (4, 2), (4, 3)]
n = 5

class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest) in edges:
			self.adj_list[src].append(dest)


def get_route(prev, i, route):
	if i >= 0:
		get_route(prev, prev[i], route)
		route.append(i)


from collections import deque
def bfs(graph, source, target, n):
	queue = deque([source])

	dist = [0] * n

	visited = [False] * n
	visited[source] = True
	prev = [-1] * n
	route = []

	while queue:
		node = queue.popleft()
		for neighbor in graph.adj_list[node]:
			
			if not visited[neighbor]:
				queue.append(neighbor)
				dist[neighbor] += 1 + dist[node]
				# print(neighbor, dist[neighbor]) #<-- helpful debugging step
				prev[neighbor] = node
				# note that you need to say at this point the neighbor has been visited to make sure you're not traversing the point again unnecessarily with a future node. 
				visited[neighbor] = True


				### if you want to stop at the specific destination ###
				# make sure you put target as an argument in the function 

				if neighbor == target:
					get_route(prev, neighbor, route)
					return f'path {source} --> {neighbor}: min cost: {dist[neighbor]}, route: {route}'

				### if you want to stop at the specific destination ###

		visited[node] = True

	# for getting the route information 
	for i in range(n):
		if i != source:
			get_route(prev, i, route)
			print(f'path {source} --> {i}: min cost: {dist[i]}, route: {route}')
			route.clear()



graph = Graph(edges, n)
# print(graph.adj_list)

# for finding shortest path from every node to every other node
print('bfs')
# for source in range(n):
# 	bfs(graph, source, n)

# for finding a specific path:
print('single path bfs', bfs(graph, 0, 3, n))





				# BFS vs Dijkstra's algorithm thoughts:

# dijkstra's is just a bfs algorithm that has a relaxation component in it (Where if checks to see if the distance or weight through the current point from the source is less than the distance it already has assigned). to do this, it uses a priority queue (min heap). they both can stop at a specific node, and both can get the minimum path and routes from any point to every other point. 

# however, bfs is not greedy, and dijkstra's is because of that relaxation step, comparing the values.

# as mentioned, bfs cannot deal with weighted graphs, as it assumes a uniform cost. for weighted graphs with exclusively positive values, dijkstra's is needed. this ability to deal with positive weights is the critical advantage of dijkstra's comapred to bfs. otherwise, with bfs, you're limited to finding the strict number of "steps" it takes to do something, but with dijkstra's, if some steps are more expensive than others, you can find the least expensive path. 


# dijkstra time: O(e * log(v))
# bfs time: O(v + e)

# side note: unlike bfs, dfs will not always return the optimal path because of the order in which it visits the next node. the stack data structure essentially requires it to make exponentially worse moves compared to bfs because it will drill down into whatever the first thing it finds is until there's nothing left, whereas bfs will proceed evenly in order of increasing distance from the source (bfs = siblings visited first, dfs = children visited first). 

# think of 3 nodes in a triangle; with bfs the optimal path is found because regardless of the starting node, the other two are explored, costing 1 step. with dfs, if you begin on the correct node, it costs 1 step, but if you start on the wrong node, it will cost 2 steps. 

# an interesting modification to dfs called iterative deepening is potentially a way to make dfs less expensive than bfs while still finding the optimal path.
# https://stackoverflow.com/questions/14784753/shortest-path-dfs-bfs-or-both

