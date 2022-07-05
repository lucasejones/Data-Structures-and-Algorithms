
'''
Introduction


This is a master file for implementing various sorting algorithms. 
After each implementation, there's an analysis of each algorithm's pros and cons, as well as its time and space complexities. 


We'll cover the following sorts: 
 
	1. Quick sort

	2. Insertion sort

	3. Selection sort

	4. Merge sort

	5. Counting sort

	6. Topological sort

'''



###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###

# 1. Quicksort
	# a. Lomuto
	# b. Hoare's partitioning
	# c. Dutch National Flag
	# d. Hybrid With Insertion

	
	
	
	
				# Implementation
###_________________________________________________________________________###
	
# a. Lomuto:

a = [9, -3, 5, 2, 6, 8, -6, 1, 3]


def quicksort(a, start = 0, end = len(a) - 1):
	if start >= end:
		return 

	pivot = partition(a, start, end)

	quicksort(a, start, pivot - 1)
	quicksort(a, pivot + 1, end)


def swap(a, i, j):
	temp = a[i]
	a[i] = a[j]
	a[j] = temp


def partition(a, start, end):
	pivot = a[end]
	p_index = start

	for i in range(start, end):
		if a[i] <= pivot:
			swap(a, i, p_index)
			p_index += 1

	swap(a, end, p_index)

	return p_index

quicksort(a)
print('Lomuto', a)




# b. Hoare's partitioning

a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]

def quick_hoare(a, start, end):
	if start >= end:
		return

	pivot = partition(a, start, end)

	quick_hoare(a, start, pivot)
	quick_hoare(a, pivot + 1, end)

def swap(a, i, j):
	temp = a[i]
	a[i] = a[j]
	a[j] = temp

def partition(a, start, end):
	pivot = a[start]
	i, j = start - 1, end + 1

	while True:

		while True:
			i += 1
			if a[i] >= pivot:
				break

		while True:
			j -= 1
			if a[j] <= pivot:
				break

		if i >= j:
			return j

		swap(a, i, j)

quick_hoare(a, 0, len(a) - 1)
print('Quick Hoare', a)




# c. Dutch National Flag
a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]

def quick_dutch(a, start, end):
	if start >= end:
		return

	if end - start == 1:
		if a[start] < a[end]:
			return
		swap(a, start, end)

	x, y = partition(a, start, end)

	quick_dutch(a, start, x)
	quick_dutch(a, y, end)

def swap(a, i, j):
	temp = a[i]
	a[i] = a[j]
	a[j] = temp

def partition(a, start, end):
	mid = start
	pivot = a[end]
	
	while mid <= end:

		if a[mid] < pivot:
			swap(a, start, mid)
			start += 1
			mid += 1

		elif a[mid] > pivot:
			swap(a, mid, end)
			end -= 1

		else:
			mid += 1

	return start - 1, mid

quick_dutch(a, 0, len(a) - 1)
print('Quick Dutch', a)




# d. Hybrid with Insertion
a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]

def quick_insert(a, start, end, cutoff):
	if start >= end:
		return

	if end - start <= cutoff:
		insertion(a, start, end)
		# print('1 done')
	else:
		pivot = partition(a, start, end)

		quick_insert(a, start, pivot, cutoff)
		quick_insert(a, pivot + 1, end, cutoff)


def insertion(a, start, end):
	for i in range(start, end + 1):
		j = i
		v = a[i]

		while j > 0 and a[j - 1] > v:
			a[j] = a[j - 1]
			j -= 1

		a[j] = v


def swap(a, i, j):
	temp = a[i]
	a[i] = a[j]
	a[j] = temp


def partition(a, start, end):
	pivot = a[start]
	i, j = start - 1, end + 1

	while True:
		while True:
			i += 1
			if a[i] >= pivot:
				break
		while True:
			j -= 1
			if a[j] <= pivot:
				break

		if i >= j:
			return j

		swap(a, i, j)


quick_insert(a, 0, len(a) - 1, 3)
print('Quick hybrid', a)



# Bonus implementation

# Here's an interesting variant to illustrate DAC aspect of quicksort, found at the link below. It's not performant, and is not in place.
# I thought it might be nice to have this as a reference.
# https://stackoverflow.com/questions/18262306/quicksort-with-python

def other_quick(nums):
	less = []
	equal = []
	more = []

	if len(nums) > 1:
		print(nums)
		pivot = nums[0]
		for x in nums:
			if x < pivot:
				less.append(x)
			elif x == pivot:
				equal.append(x)
			else:
				more.append(x)
		return other_quick(less) + equal + other_quick(more)
	return nums

# print(other_quick(nums))
# print(nums)

# this is not in place, so it's worse on memory than standard quicksort. it also chooses a very poor pivot. However, it demonstrates divide and conquer very well as a conceptual exercise.
	# values relative to the pivot are shuttled into their respective lists. each list is then recursively put through the same process until there's only 1 element in it, and combined with the other elements.   

	
	
				# Analysis
###_________________________________________________________________________###

# pros and cons:
	# pros: 
		# in-place (a pro if this is desired)
		# generally seen as the fastest sort for large inputs (specifically better than merge sort)
		# can be modified to handle a wide range of cases (repeating elements, hybrid with insertion sort)

	# cons:
		# not a stable sort
		# if not handled approrpiately, repeating elements or poorly selected pivots will render it inefficient
		# not ideal for smaller inputs due to the number of swaps needed.

	# neutral: 
		# is a comparison sort, and is divide-and-conquer.

# when to use:
	# you have something you need sorted (it should be the default choice)
	# you have a large input 
	# you're willing to try out variations to suit that input (repeating elements, insertion hybrid)
	# you don't mind an unstable sort
	# you would like to save some memory 


# complexity analysis:
	# time: O(n log n), but worst case of O(n^2). best case is if pivot roughly bisects each partition. 
	# space: O(n) for the call stack


###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###









###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###

# Insertion Sort


				# Implementation
###_________________________________________________________________________###

a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]

def insertion_sort(a):
	for i, v in enumerate(a):
		j = i

		while j > 0 and a[j - 1] > v:
			a[j] = a[j - 1]
			j -= 1

		a[j] = v

insertion_sort(a)
print('Insertion', a)



				# Analysis
###_________________________________________________________________________###


# pros and cons:
	# pros: 
		# in-place (if this is seen as a pro)
		# stable
		# can integrate new information during the sort easily, allowing for dynamic usage, unlike many other algorithms. 
		# space-efficient 
		# can be used effectively for small arrays, making it useful for hybrid sort algorithms (as in quicksort)
		# performs fewer comparisons than selection sort because not all remaining elements need to be scanned.
		

	# cons:
		# very slow for reversed arrays or large, poorly sorted arrays


	# neutral: 
		# is a comparison sort
		# keeps track of two subsets: sorted and unsorted. it goes from left to right through the unsorted subset to fnid the next unsorted value and compares it against all other values in the sorted subset (moving backwards) until it finds the correct position and places it there.

# when to use:
	# you'd like an in-place, stable sort that works particualrly well on small arrays. 
	# it's especially useful in contexts where the input is already mostly sorted.
	# if you have dynamic data (incoming information that needs continual integration into the sort)
	# if you'd like to incorporate it in a hybrid architecture with another sorting algorithm
	


# complexity analysis:
	# time: O(n^2) worst case (a descending sorted list), best case approaches O(n) (mostly sorted list)
	# space: O(1) (though many swaps are required)


###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###









###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###

# Selection Sort

				# Implementation
###_________________________________________________________________________###

a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]

def selection_sort(a):
	for i in range(len(a) - 1):
		smallest = i

		for j in range(i + 1, len(a)):
			if a[j] < a[smallest]:
				smallest = j

		swap(a, smallest, i)

def swap(a, i, j):
	temp = a[i]
	a[i] = a[j]
	a[j] = temp


selection_sort(a)
print('Selection', a)



				# Analysis
###_________________________________________________________________________###

# pros and cons:
	# pros: 
		# performs few write operations, so if memory space is limited, this is a good choice compared to insertion sort 
		# in-place sort (if you like this)
		# it's good for checking to see if a list is sorted
		# generally outperforms bubble sort


	# cons:
		# generally worse than insertion sort
		# O(n^2) time complexity is quite bad
		# is an unstable sort
		

	# when to use:
		# you're ok with O(n^2) time complexity (such as with relatively small lists) but are prioritizing a minimal amount of swaps to optimize memory. Because it will perform at most n swaps compared to insertion sort's n^2 swaps (worst case), it can be the better choice in this context.
	


# complexity analysis:
	# time: O(n^2) because for each element you have to look through all unsorted elements to find the smallest one. 
	# space: O(1) because no additional space is required. Once the swap occurs to move the next smallest element to the correct position, no future swaps at that location need to occur. 


###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###









###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###

# Merge sort
	# a. recursive top-down
	# b. iterative bottom-up

	
				# Implementation
###_________________________________________________________________________###	
	
	
	
# a. Recursive Top-down

a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]
a_aux = a.copy()

def merge_sort_rec(a, aux, low, high):

	if high == low:
		return

	mid = low + (high - low) // 2

	merge_sort_rec(a, aux, low, mid)
	merge_sort_rec(a, aux, mid + 1, high)

	merge(a, aux, low, mid, high)


def merge(a, aux, low, mid, high):

	i = low
	k = low
	j = mid + 1

	while i <= mid and j <= high:
		if a[i] <= a[j]:
			aux[k] = a[i]
			i += 1
		else:
			aux[k] = a[j]
			j += 1
		k += 1

	while i <= mid:
		aux[k] = a[i]
		i += 1
		k += 1

	for i in range(low, high + 1):
		a[i] = aux[i]


merge_sort_rec(a, a_aux, 0, len(a) - 1)
print('Merge Recursive Top-down', a)



# b. Iterative Bottom-up
a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]

def merge_sort(a):
	start, end = 0, len(a) - 1
	a_aux = a.copy()

	m = 1
	while m <= end - start:
		for i in range(start, end, m * 2):
			low = i
			mid = i + m - 1
			high = min(i + (m * 2) - 1, end)
			merge(a, a_aux, low, mid, high)
		m *= 2


def merge(a, aux, low, mid, high):
	i = low
	k = low
	j = mid + 1

	while i <= mid and j <= high:
		if a[i] <= a[j]:
			aux[k] = a[i]
			i += 1
		else:
			aux[k] = a[j]
			j += 1
		k += 1

	while i <= mid:
		aux[k] = a[i]
		i += 1
		k += 1

	for i in range(low, high + 1):
		a[i] = aux[i]

merge_sort(a)
print('Merge Iterative Bottom-up', a)



				# Analysis
###_________________________________________________________________________###

# pros and cons:
	# pros: 
		# is stable (a pro if this is desired)
		# n * log n time complexity is decent and guaranteed
		# performs well on any orderable sequence of integers 
		

	# cons:
		# is out-of-place, which takes additional memory (more relevant for recursive option than iterative which holds only < n at any given time)
		# slower than its competitor, quicksort

	
	# neutral: 
		# it's a comparative, divide and conquer algorithm 
		

# when to use:
	# you want a guaranteed run time of O(n * log n)
	# you want a stable sort
	# is useful in parallelization 
	


# complexity analysis:
	# time: O(n * log n), each element is visited (n) and the list is recursively split into equal-length sublists (log n)
	# space: O(n) for the call stack 


###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###









###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###

# Counting Sort
	# a. Method #1
	# b. Method #2


				# Implementation
###_________________________________________________________________________###	

# a. Method #1
	
a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]
k = 6

def counting_sort(a, k):
	output = [0] * len(a)
	freq = [0] * (k + 1)

	for i in a:
		freq[i] += 1

	total = 0
	for i in range(1, k + 1):
		old_count = freq[i]
		freq[i] = total
		total += old_count

	for i in a:
		output[freq[i]] = i
		freq[i] += 1

	for i in range(len(a)):
		a[i] = output[i]


counting_sort(a, k)
print('Counting Method 1', a)




# b. Method #2: this is maybe more common, but slower than the above.
a = [4, 2, 6, 1, 2, 3, 5, 6, 2, 1, 4, 3, 2]

def counting_alt(a):
	size = len(a)
	output = [0] * size
	count = [0] * (max(a) + 1)

	for i in range(0, size):
		count[a[i]] += 1

	for i in range(1, max(a) + 1):
		count[i] += count[i - 1]

	i = size - 1
	while i >= 0:
		output[count[a[i]] - 1] = a[i]
		count[a[i]] -= 1
		i -= 1

	for i in range(0, size):
		a[i] = output[i]

counting_alt(a)
print('Counting Method 2', a)



				# Analysis
###_________________________________________________________________________###

# pros and cons:
	# pros: 
		# it is a stable sort (though you can modify it to be unstable)
		# it is an in-place sort, if you view this as a pro (regarding space it certainly is)
		# O(n + k) is a very fast sort - if you need a linear complexity, this is a good choice.
		# it can be very space-efficient for small arrays with appropriate k values
		# it can be used as a subroutine in radix sort

	# cons:
		# if k is large relative to n, it is incredibly space-inefficient
		# if you need an out-of-place sort, this is not the one to choose. 

	# neutral: 
		# it is not a comparison sort, nor divide and conquer - it works by computing the prefix sum (adding up the integers of the array) and uses their frequencies and totals to determine the correct order.

# when to use:
	# you want a linear-time, stable, in-place sort for a small array where the maximum value is not significantly larger than the length of that array.


# complexity analysis:
	# time: O(n + k) worst case average and best case.
		# O(n) + O(k) + O(n) + O(k) = O(n + k)
		# better if k is substantially smaller than n where n is the number of elements. in other words, if a value in a small array is large, do not use counting sort. 
	# space: O(n + k) because it uses auxiliary arrays of size n and size k.


# be able to talk through this one in particular.


###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###









###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###

# Topological Sort
	# a. Kahn's algorithm
	# b. DFS

	
				# Implementation
###_________________________________________________________________________###	
	
# a. Kahn's:

class Graph:
	in_degree = None

	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		self.in_degree = [0] * n

		for (src, dest) in edges:
			self.adj_list[src].append(dest)
			self.in_degree[dest] += 1

edges = [(0, 3), (1, 2), (3, 2)]
n = 4

grapht = Graph(edges, n)
# print(graph.adj_list, graph.in_degree)

from collections import deque

def topological_sort(graph, n):
	in_degree = grapht.in_degree
	topo_sort = []
	no_incoming = deque([i for i in range(n) if in_degree[i] == 0])

	while no_incoming:
		vertex = no_incoming.pop()
		topo_sort.append(vertex)

		for m in grapht.adj_list[vertex]:
			in_degree[m] -= 1
			if in_degree[m] == 0:
				no_incoming.append(m)

	for i in range(n):
		if in_degree[i]:
			return None
	
	return topo_sort

topo_sorted = topological_sort(grapht, n)
if topo_sorted:
	print('Topological Kahn', topo_sorted)
else:
	print('At least 1 cycle present, topologicial sort not possible.')


	

# b. DFS

class Graph:
	def __init__(self, edges, n):
		self.adj_list = [[] for _ in range(n)]
		for (src, dest) in edges:
			self.adj_list[src].append(dest)

edges = [(0, 3), (1, 2), (3, 2)]
n = 4
graph = Graph(edges, n)
# print(graph.adj_list)


def topological_dfs(graph, n):
	output = []
	departure = [-1] * n
	discovered = [False] * n
	time = 0

	for i in range(n):
		if not discovered[i]:
			time = dfs(graph, i, discovered, departure, time)

	for i in reversed(range(n)):
		output.append(departure[i])

	return output

def dfs(graph, v, discovered, departure, time):
	discovered[v] = True

	for u in graph.adj_list[v]:
		if not discovered[u]:
			time = dfs(graph, u, discovered, departure, time)

	departure[time] = v
	time += 1

	return time


dfs_sorted = topological_dfs(graph, n)
print('Topological DFS', dfs_sorted)



				# Analysis
###_________________________________________________________________________###

# pros and cons:
	# pros: 
		# linear time
		# very little space (linear) required (especially with BFS)
		# BFS and DFS options

	# cons:
		# selection of BFS vs DFS is somewhat arbitrary
		# potentially complicated code can cause confusion in a team
		

	# neutral: 
		# BFS thinking: 
			# identify any nodes with no incoming edges. sequentially remove outgoing edges from these nodes, separating them from the graph. Should this process reveal nodes that now also have no incoming edges, repeat the process until all nodes have been addressed. if any nodes remain with incoming edges, it is a cyclic graph.
		# DFS thinking:
			# keep track of which nodes have been visited, and output the nodes in order of decreasing departure times. Because a cyclic graph will have a back edge, an acylic graph can be sorted in this manner. for any given node, mark it as visited, and see if there are adjacent nodes. If so, recursively repeat this process until there are no neighbors, and increment the depature time. Eventually you will have all nodes and their departure times, which in reverse order reveals the ascending sort.

# when to use:
	# you need to determine a valid order to proceed through steps in a process; ensuring that all prerequisites are met before continuing. (instruction scheduling, deciding in which order to load tables with foreign keys into a database, resolving data dependencies, etc)
	# if you're working with a graph.
	# BFS (kahn) if you want to use a queue, DFS if you'd like to use a stack via recursion.
	# if you need to detect whether a graph is cyclic.


# complexity analysis:
	# time: 
		# O(v + e) in all cases. it takes linear time to visit each node and edge.
	# space:
		# O(v). The queue stores the vertices.


###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###




# Super-condensed Factblast

# quick
	# time: O(n * log n) most times, O(n^2) if poorly implemented
	# space: O(n)
	# in-place, unstable, great for large inputs, especially if used with insertion

# merge
	# time: O(n * log n) in all cases
	# space: O(n)
	# stable, out-of-place, guaranteed O(n * log n), good for paralellization

# counting
	# time: O(n + k) all cases
	# space: O(n + k) for each extra array
	# linear time, fast, stable, efficient, in-place

# selection
	# time: O(n^2) 
	# space: O(1) (at most n swaps)
	# quadratic time, slow, unstable, memory-optimized, in-place

# insertion
	# time: O(n^2) worst, O(n) best
	# space: O(1)
	# stable, in-place, dynamic data sorting, space-efficient, fast in certain cases

# topological
	# time: O(v + e)
	# space: O(v) (plus call stack if dfs)
	# for determining valid order of operations given prerequisites



