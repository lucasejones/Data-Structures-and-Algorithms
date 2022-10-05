


# Types of problems to test:

# 1. linked list problems
	# create a singly linked list from scratch
	# create a doubly linked list from scratch

# 2. ring buffer
	# create a ring buffer from scratch

# binary tree problems
	# create algos for iterative and recursive solutions for all 3 DFS methods
	# create iterative for BFS
	# create iterative and recursive algo for level-by-level 

# binary search problems
	# find the minimum in a rotated sorted list

###___________________________________________________________________###
###___________________________________________________________________###
###___________________________________________________________________###






##### linked list problems #####
#______________________________#

	# 1. create a singly linked list from scratch

# a linked list class will involve the init function, a separate listnode class, and functions to: add at a given index, delete at a given index, get from an index, add to head, and add to tail. 


class list_node:
	def __init__(self, x):
		self.val = x
		self.next = None


class linked_list:
	def __init__(self):
		self.head = list_node(0)
		self.size = 0


	def add_at_index(self, index, val):
		if index < 0:
			index = 0 

		if index > self.size:
			return

		pred = self.head
		for i in range(index):
			pred = pred.next

		self.size += 1
		new_node = list_node(val)
		new_node.next = pred.next
		pred.next = new_node


	def delete_at_index(self, index):
		if index < 0 or index >= self.size:
			return -1

		pred = self.head
		for i in range(index):
			pred = pred.next

		self.size -= 1

		pred.next = pred.next.next


	def get(self, index):
		if index < 0:
			index = 0 

		if index >= self.size:
			return 

		current = self.head
		for i in range(index + 1):
			current = current.next

		return current.val


	def add_to_head(self, val):
		return self.add_at_index(0, val)


	def add_to_tail(self, val):
		return self.add_at_index(self.size, val)


# my_linked_list = linked_list()
# print(my_linked_list)
# print(my_linked_list.add_at_index(0, 10))
# print(my_linked_list.get(0))
# print(my_linked_list.add_to_head(200))
# print(my_linked_list.get(0))
# print(my_linked_list.delete_at_index(1))
# print(my_linked_list.get(0))
# print(my_linked_list.add_to_tail(300))
# print(my_linked_list.get(1))

# time and space complexity analysis:
# time: O(1) -> O(n) for adding or deleting to head, worse as you add to tail.
#		O(1) for get.  
# space: O(n), where n is number of nodes. 1 pointer per node. 

# remember: 
# the if statement for the starting conditions in each function.
# if index < 0 or index >= self.size  <--- this applies to get, and delete.
# for add, you must allow index to = self.size, because the first node will be added to a size of 0.






	# 2. create a doubly linked list from scratch

# same as singly, but with additional coding to create a second pointer to allow reverse traversal. 


class double_list_node:
	def __init__(self, x):
		self.val = x
		self.next, self.prev = None, None



class doubly_linked_list:
	def __init__(self):
		self.size = 0
		self.head, self.tail = double_list_node(0), double_list_node(0)
		self.head.next = self.tail
		self.tail.prev = self.head


	def add_at_index(self, index, val):
		if index < 0:
			index = 0

		if index > self.size:
			return

		if index < self.size - index:
			pred = self.head
			for i in range(index):
				pred = pred.next
			succ = pred.next
		else:
			succ = self.tail
			for i in range(self.size - index):
				print(succ.val)
				succ = succ.prev
			pred = succ.prev

		self.size += 1

		new_node = double_list_node(val)
		new_node.next = succ
		new_node.prev = pred
		pred.next = new_node
		succ.prev = new_node

# [1, 2, 3, 4, 5, 6, 7]

	def delete_at_index(self, index):
		if index < 0 or index >= self.size:
			return -1

		if index < self.size - index:
			pred = self.head
			for i in range(index):
				pred = pred.next
			succ = pred.next.next
		else:
			succ = self.tail
			for i in range(self.size - index - 1):
				succ = succ.prev
			pred = succ.prev.prev

		self.size -= 1

		pred.next = succ
		succ.prev = pred




	def get(self, index):
		if index < 0 or index >= self.size:
			return -1

		if index < self.size - index:
			current = self.head
			for i in range(index + 1):
				current = current.next
		else:
			current = self.tail
			for i in range(self.size - index):
				current = current.prev

		return current.val



	def add_to_head(self, val):
		return self.add_at_index(0, val)


	def add_to_tail(self, val):
		return self.add_at_index(self.size, val)




my_doubly_linked_list = doubly_linked_list()
print(my_doubly_linked_list)
print(my_doubly_linked_list.add_to_head(100))
print(my_doubly_linked_list.get(0))
print(my_doubly_linked_list.add_to_tail(500))
print(my_doubly_linked_list.add_at_index(1, 300))
print(my_doubly_linked_list.get(0), my_doubly_linked_list.get(1), my_doubly_linked_list.get(2))

print(my_doubly_linked_list.delete_at_index(1))

print(my_doubly_linked_list.get(0), my_doubly_linked_list.get(1), my_doubly_linked_list.get(2))

print(my_doubly_linked_list.add_at_index(1, 400))

print(my_doubly_linked_list.get(0), my_doubly_linked_list.get(1), my_doubly_linked_list.get(2))




# remember: 
	# in the get function, when finding the current value, if walking backwards from the tail, you need self.size - index - 1. otherwise, you'll get the predecessor and not the current. the - 1, because you're coming from the end, adds 1 to the index, just like the relationship between pred and current is range(index) and range(index + 1).
	# the reason that when coming from the tail, the if conditions are different (range(self.size - index) vs range(self.size - index - 1)) for add versus delete is:
		# if you're adding, you want to identify the adjacent numbers, to scoot one inbetween, which will replace the index of the succ. this succ is found with self.size - index. 
		# if you're deleting, you want to identify the numbers on either side of the index you're deleting. this succ is found with self.size - index - 1. 

	# when finding pred and succ, you are indeed finding the node directly before or directly after the index you specify. for i in range(index) gives you, starting from the pseudo-head/tail, index iterations.
	# ex: [1, 2, 3, 4, 5, 6, 7]
	# to delete index 2, pred and succ will be found to be 2 and 4. 
	# this is because for i in range(2) starting from the pseudo head gives you: 
		# 1st iteration: pseudo-head "0" turns into head.next, which is 1.
		# 2nd iteration: 1 turns into 1.next, which is 2.
		# end of loop. 



# time and space complexity analysis:
# time: adding or deleting to head or tail: O(1)
#		adding, deleting, or getting from any other index:	
#			O(min(k, n - k)) at worst where k is the index and n is number of nodes.

# space: O(n), n is number of nodes, with 2 pointers per node. 





###___________________________________________________________________###
###___________________________________________________________________###




##### ring bufffer #####
#______________________________#

	# 1. create a ring buffer from scratch



class circular_queue:
	def __init__(self, size):
		self.queue = [0] * size
		self.size = size
		self.head_index = 0
		self.count = 0


	def enqueue(self, val):
		if self.is_full():
			return False

		self.queue[(self.head_index + self.count) % self.size] = val
		self.count += 1
		return True


	def dequeue(self):
		if self.is_empty():
			return False

		# self.queue[self.head_index] = 0
		self.head_index = (self.head_index + 1) % self.size
		self.count -= 1
		return True


	def front(self):
		if self.is_empty():
			return False

		return self.queue[self.head_index]


	def rear(self):
		if self.is_empty():
			return False

		return self.queue[(self.head_index + self.count - 1) % self.size]


	def is_full(self):
		return self.count == self.size


	def is_empty(self):
		return self.count == 0


	def contents(self):
		return self.queue



my_circ = circular_queue(5)
print(my_circ)
print(my_circ.enqueue(10))
print(my_circ.enqueue(20))
print(my_circ.enqueue(30))
print(my_circ.enqueue(40))
print(my_circ.enqueue(50))
print(my_circ.enqueue(60))
print(my_circ.contents())
print(my_circ.dequeue())
print(my_circ.enqueue(100))
print(my_circ.front())
print(my_circ.rear())
print(my_circ.is_full())
print(my_circ.is_empty())
print(my_circ.contents())


# remember: 
	# the operations you can do on a circular queue are:
		# enqueue, dequeue, front, rear, is_full, and is_empty.
		# you can also add a contents function to see what's in your queue.

	# for dequeue, all you need to do is move the head index up by one. don't use the self.queue[] syntax for this. 

	# don't forget to increment or decrement the count whenever you're enqueing or dequeing

	# is_full relies on the compariso between self.count and self.size. 

# complexity analysis:
# time complexity: O(1) for all operations (there is no loop anywhere)
# space complexity: O(n) where n is the number of elements














###___________________________________________________________________###
###___________________________________________________________________###



##### binary tree problems #####
#______________________________#


# driver code for binary tree problems

class tree_node:
	def __init__(self, x):
		self.val = x
		self.left, self.right = None, None

root = tree_node(1)
root.left = tree_node(2)
root.right = tree_node(3)
root.right.left = tree_node(4)
root.right.right = tree_node(5)

# this tree looks like this:
# 		1 
# 	   / \ 
# 	  2   3
#        / \ 
#       4   5 


	# 1. create algos for iterative and recursive solutions for all 3 DFS methods


	# pre-order iterative (root, left, right)
def pre_order(root):
	if root == None:
		return []

	stack = [root]
	output = []

	while stack:
		root = stack.pop()
		if root:
			output.append(root.val)
			stack.append(root.right)
			stack.append(root.left)

	return output

print(pre_order(root))


	# pre-order recursive

def pre_order(root):
	output = []
	dfs_pre(root, output)
	return output

def dfs_pre(root, output):
	if root:
		output.append(root.val)
		dfs_pre(root.left, output)
		dfs_pre(root.right, output)

print(pre_order(root))




	# in-order iterative

def in_order(root):
	if root == None:
		return []

	stack = []
	output = []
	current = root

	while stack or current:
		if current:
			stack.append(current)
			current = current.left
		else:
			current = stack.pop()
			output.append(current.val)
			current = current.right
	return output

print(in_order(root))



	# in-order recursive

def in_rec(root):
	output = []
	dfs_in(root, output)
	return output

def dfs_in(root, output):
	if root:
		dfs_in(root.left, output)
		output.append(root.val)
		dfs_in(root.right, output)

print(in_rec(root))




	# post-order iterative

def post_order(root):
	if root == None:
		return []

	stack = [root] * 2
	output = []

	while stack:
		current = stack.pop()
		if stack and stack[-1] == current:
			if current.right:
				stack += [current.right] * 2
			if current.left:
				stack += [current.left] * 2
		else:
			output.append(current.val)

	return output


print(post_order(root))



	# post-order recursive

def post_rec(root):
	output = []
	dfs_post(root, output)
	return output

def dfs_post(root, output):
	if root:
		dfs_post(root.left, output)
		dfs_post(root.right, output)
		output.append(root.val)

print(post_rec(root))






	# 2. create iterative for BFS

from collections import deque

def bfs(root):
	if root == None:
		return []

	queue = deque([root])
	output = []

	while queue:
		root = queue.popleft()
		if root:
			output.append(root.val)
			queue.append(root.left)
			queue.append(root.right)

	return output


print(bfs(root))





	# 3. create iterative and recursive algo for level-by-level 

# iterative (with zigzag)
def lev_by_lev(root):
	if root == None:
		return []

	queue = deque([root])
	output = []
	is_lev_odd = False

	while queue:
		level = []
		level_size = len(queue)

		for i in range(level_size):
			root = queue.popleft()
			level.append(root.val)

			if root.left:
				queue.append(root.left)
			if root.right:
				queue.append(root.right)

		if is_lev_odd:
			level.reverse()

		output.append(level)

		is_lev_odd = not is_lev_odd


	return output

print(lev_by_lev(root))



# recursive
def lev_by_rec(root):
	if root == None:
		return []

	output = []

	def helper(root, level):
		if len(output) == level:
			output.append([])

		output[level].append(root.val)

		if root.left:
			helper(root.left, level + 1)
		if root.right:
			helper(root.right, level + 1)


	helper(root, 0)
	return output


print(lev_by_rec(root))





print('super mega deluxe master class practice')
print('_______________________________________')



# pre-order iterative
def pre_order(root):
	if root == None:
		return []

	stack = [root]
	output = []

	while stack:
		root = stack.pop()
		if root:
			output.append(root.val)
			stack.append(root.right)
			stack.append(root.left)
	return output

print(pre_order(root))


# pre-order recursive
def pre_rec(root):
	output = []
	dfs_pre(root, output)
	return output

def dfs_pre(root, output):
	if root:
		output.append(root.val)
		dfs_pre(root.left, output)
		dfs_pre(root.right, output)

print(pre_rec(root))



# in-order iterative

def in_order(root):
	if root == None:
		return []

	stack = []
	output = []
	current = root

	while stack or current:
		if current:
			stack.append(current)
			current = current.left
		else:
			current = stack.pop()
			output.append(current.val)
			current = current.right
	return output

print(in_order(root))



# in-order recursive
def in_rec(root):
	output = []
	dfs_in(root, output)
	return output

def dfs_in(root, output):
	if root:
		dfs_in(root.left, output)
		output.append(root.val)
		dfs_in(root.right, output)

print(in_rec(root))



# post-order iterative

def post_order(root):
	if root == None:
		return []

	stack = [root] * 2
	output = []

	while stack:
		current = stack.pop()
		if stack and stack[-1] == current:
			if current.right:
				stack += [current.right] * 2
			if current.left:
				stack += [current.left] * 2
		else:
			output.append(current.val)
	return output

print(post_order(root))



# post-order recursive
def post_rec(root):
	output = []
	dfs_post(root, output)
	return output

def dfs_post(root, output):
	if root:
		dfs_post(root.left, output)
		dfs_post(root.right, output)
		output.append(root.val)

print(post_rec(root))




# bfs iterative

from collections import deque

def bfs(root):
	if root == None:
		return []

	queue = deque([root])
	output = []

	while queue:
		root = queue.popleft()
		if root:
			output.append(root.val)
			queue.append(root.left)
			queue.append(root.right)

	return output

print(bfs(root))


# bfs level by level iterative + zig_zag

def level_by_level(root):
	if root == None:
		return []

	queue = deque([root])
	output = []
	is_level_odd = False

	while queue:
		level = []
		level_size = len(queue)

		for i in range(level_size):
			root = queue.popleft()
			level.append(root.val)

			if root.left:
				queue.append(root.left)
			if root.right:
				queue.append(root.right)

		if is_level_odd:
			level.reverse()

		output.append(level)

		is_level_odd = not is_level_odd

	return output

print(level_by_level(root))



# bfs level by level recursive

# preferred way: here's how to extract out the helper function
def lev_by_rec(root):
	if root == None:
		return []

	output = []
	helper(root, 0, output)

	return output

def helper(node, level, output):
	if len(output) == level:
		output.append([])

	output[level].append(node.val)

	if node.left:
		helper(node.left, level + 1, output)
	if node.right:
		helper(node.right, level + 1, output)

	
print(lev_by_rec(root))



# old alternate way:
def lev_by_rec(root):
	if root == None:
		return []

	output = []

	def helper(node, level):
		if len(output) == level:
			output.append([])

		output[level].append(node.val)

		if node.left:
			helper(node.left, level + 1)
		if node.right:
			helper(node.right, level + 1)

	helper(root, 0)
	return output

print(lev_by_rec(root))













###___________________________________________________________________###
###___________________________________________________________________###


##### binary search problems #####
#______________________________#

	# find the minimum in a rotated sorted list


nums = [3, 4, 5, 6, 0, 1, 2]


def min_sort_rot(nums):
	left, right = 0, len(nums) - 1

	if nums[right] > nums[left]:
		return nums[0]

	while left < right:
		pivot = left + (right - left) // 2
		# print(left, pivot, right)
		if nums[pivot] > nums[right]:
			left = pivot + 1
		else:
			right = pivot
			
	# print(left, pivot, right)
	return left

print(min_sort_rot(nums))







###___________________________________________________________________###
###___________________________________________________________________###




def pre_order(root):
	if root == None:
		return []

	stack = [root]
	output = []

	while stack:
		root = stack.pop()
		if root:
			output.append(root.val)
			stack.append(root.right)
			stack.append(root.left)
	return output

print(pre_order(root))


def pre_rec(root):
	output = []
	dfs_pre(root, output)
	return output

def dfs_pre(root, output):
	if root:
		output.append(root.val)
		dfs_pre(root.left, output)
		dfs_pre(root.right, output)

print(pre_rec(root))



def in_order(root):
	if root == None:
		return []

	stack = []
	output = []
	current = root

	while stack or current:
		if current:
			stack.append(current)
			current = current.left
		else:
			current = stack.pop()
			output.append(current.val)
			current = current.right
	return output

print(in_order(root))


def in_rec(root):
	output = []
	dfs_in(root, output)
	return output

def dfs_in(root, output):
	if root:
		dfs_in(root.left, output)
		output.append(root.val)
		dfs_in(root.right, output)

print(in_rec(root))


def post_order(root):
	if root == None:
		return []

	stack = [root] * 2
	output = []

	while stack:
		current = stack.pop()
		if stack and stack[-1] == current:
			if current.right:
				stack += [current.right] * 2
			if current.left:
				stack += [current.left] * 2
		else:
			output.append(current.val)
	return output

print(post_order(root))


def post_rec(root):
	output = []
	dfs_post(root, output)
	return output

def dfs_post(root, output):
	if root:
		dfs_post(root.left, output)
		dfs_post(root.right, output)
		output.append(root.val)

print(post_rec(root))



from collections import deque

def bfs(root):
	if root == None:
		return []

	queue = deque([root])
	output = []

	while queue:
		root = queue.popleft()
		if root:
			output.append(root.val)
			queue.append(root.left)
			queue.append(root.right)
	return output

print(bfs(root))


def level_by_level(root):
	if root == None:
		return []

	queue = deque([root])
	output = []
	is_level_odd = False

	while queue:
		level = []
		level_size = len(queue)

		for i in range(level_size):
			root = queue.popleft()
			level.append(root.val)

			if root.left:
				queue.append(root.left)
			if root.right:
				queue.append(root.right)

		if is_level_odd:
			level.reverse()
		is_level_odd = not is_level_odd

		output.append(level)

	return output

print(level_by_level(root))


def bfs_rec(root):
	if root == None:
		return []

	output = []
	helper(root, 0, output)

	return output

def helper(node, level, output):
	if len(output) == level:
		output.append([])

	output[level].append(node.val)

	if node.left:
		helper(node.left, level + 1, output)
	if node.right:
		helper(node.right, level + 1, output)

print(bfs_rec(root))






















