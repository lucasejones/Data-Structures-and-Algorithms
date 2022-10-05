# 3 main applications of heap:

	# 1. heap sort
	# 2. the top-k problem
	# 3. the k-th element




#__________________________________________________________________________#
#__________________________________________________________________________#
#__________________________________________________________________________#
								# Heap Sort
#__________________________________________________________________________#
#__________________________________________________________________________#
#__________________________________________________________________________#

# 1. heapify all elements into a heap
# 2. record and delete the top element
# 3. put the top element into a new array T that store all sorted elements. 
# 4. repeat 2 and 3 until heap is empty. T now contains the sorted elements.

nums = [5, 6, 2, 4, 9, 1, 7, 8]

import heapq

nums = [5, 6, 2, 4, 9, 1, 7, 8]

# this version is out of place (maintains original list)
def heap_sort(a):
	t = []
	for i in a:
		heapq.heappush(t, i)

	return [heapq.heappop(t) for i in range(len(t))]

out_place = heap_sort(nums)
print(out_place)



nums = [5, 6, 2, 4, 9, 1, 7, 8]
# this version modifies in place
def heap_sort(a):
	heapq.heapify(a)
	return [heapq.heappop(a) for i in range(len(a))]

in_place = heap_sort(nums)
print(in_place)



# max heap sort (descending heap sort)
nums = [5, 6, 2, 4, 9, 1, 7, 8]

def heap_max(a):
	t = []
	for i in a:
		heapq.heappush(t, -1 * i)

	return [heapq.heappop(t) * -1 for i in range(len(t))]


print(heap_max(nums))


# complexity analysis
	# time: O(n log n) - each element gets popped from the heap (n) and the subsequent heapify process is log n.
	# space: O(n) - the heap takes n space





#__________________________________________________________________________#
#__________________________________________________________________________#
#__________________________________________________________________________#
								# The Top K Problem
#__________________________________________________________________________#
#__________________________________________________________________________#
#__________________________________________________________________________#

# naive approach:
	# sort the list and return the -k index
		# ex: return sorted(input_list)[-k]

	# this gives time of O(n * log n), space of O(1).
	# we can instead improve the time by sacrificing a little bit of space. 


# approach 1:
	# 1. make a min (or max) heap
	# 2. add all elements of the input into the heap
	# 3. pop the top element off and add it to the output list
	# 4. repeat step 3 until k elements have been added


# min heap
nums = [5, 6, 2, 9, 4, 1, 4, 2, 5]
import heapq

def heap_sort(a, k):
	t = []
	heapq.heapify(a)

	for i in range(k):
		t.append(heapq.heappop(a))

	return t

u = heap_sort(nums, 3)
print(u)


# max heap
nums = [5, 6, 2, 9, 4, 1, 4, 2, 5]
import heapq

def heap_sort(a, k):
	max_heap = [-i for i in a]
	heapq.heapify(max_heap)
	t = []

	for i in range(k):
		t.append(heapq.heappop(max_heap) * -1)

	return t


u = heap_sort(nums, 3)
print(u)


# more condensed version of min_heap above
nums = [5, 6, 2, 9, 4, 1, 4, 2, 5]
import heapq

def heap_sort(a, k):
	heapq.heapify(a)

	return [heapq.heappop(a) for i in range(k)]

u = heap_sort(nums, 3)
print(u)


# complexity analysis:
	# time: O(k * log n + n)
		# constructing the heap takes O(n)
		# each element removed from heap takes O(log n) 
		# this O(log n) removal process occurs k times.
		# O(k elements * removal + heap construction)
	# space: O(n)
		# the heap stores n elements





# approach 2:
	# 1. create a min heap of size k
	# 2. heapify it with negative values from a
	# 3. for the remaining values, compare against the top position
		# if the negative value is larger than the top position, replace the top value with that negative value 
	# 4. re-convert the signs the heap and return the values

nums = [5, 6, 2, 9, 4, 1, 4, 2, 5]

import heapq

def k_smallest_built(a, k):
	return heapq.nsmallest(k, a)
	# return heapq.nlargest(k, a)

r = k_smallest_built(nums, 4)
# print(r)


def k_largest_built(a, k):
	# return heapq.nsmallest(k, a)
	return heapq.nlargest(k, a)

r = k_largest_built(nums, 4)
# print(r)


nums = [5, 6, 2, 9, 4, 1, 4, 2, 5]
def k_smallest(a, k):
	heap = [-a[i] for i in range(k)]
	heapq.heapify(heap)
	# print(heap)

	for i in range(k, len(a)):
		# print(a[i], heap[0])
		if -a[i] > heap[0]:
			# print('did it', heap)
			heapq.heappushpop(heap, -a[i])
			# print(heap)

	return [-i for i in heap]

r = k_smallest(nums, 4)
# print(r)



nums = [5, 6, 2, 9, 4, 1, 4, 2, 5]

def k_largest(a, k):
	heap = [a[i] for i in range(k)]
	heapq.heapify(heap)
	# print(heap)

	for i in range(k, len(a)):
		if a[i] > heap[0]:
			heapq.heappushpop(heap, a[i])

	return heap


r = k_largest(nums, 4)
# print(r)

# complexity analysis
	# time: O(n * log k)
		# O((n - k) * log k + k * log k) = O(n * log k). (n - k) is the most times the log k replacement behavior occurs. 
	# space: O(k): heap contains at most k elements



# Performance Comparison between using the built-in function and explicitly writing the logic
from time import perf_counter

start = perf_counter()
print(k_smallest_built(nums, 4))
end = perf_counter()
print('small built:', f'{end - start:.8f}')

start = perf_counter()
print(k_largest_built(nums, 4))
end = perf_counter()
print('large built:', f'{end - start:.8f}')

start = perf_counter()
print(k_smallest(nums, 4))
end = perf_counter()
print('small:', f'{end - start:.8f}')

start = perf_counter()
print(k_largest(nums, 4))
end = perf_counter()
print('large:', f'{end - start:.8f}')

# the built in is slightly worse than writing it out myself. 
# in the k_smallest function (which relies on a max heap), the extra conversion of the signs requires one more iteration through the heap which makes it a bit slower than the k_largest function (which relies on a min heap)


