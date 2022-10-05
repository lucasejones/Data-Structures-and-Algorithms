

preorder = [3, 9, 20, 15, 7]
inorder = [9, 3, 15, 20, 7]
postorder = [9, 15, 7, 20, 3]


class tree_node:
	def __init__(self, x):
		self.val = x
		self.left, self.right = None, None


root = tree_node(3)
root.left = tree_node(9)
root.right = tree_node(20)
root.left.left = tree_node(200)
root.left.right = tree_node(250)
root.right.left = tree_node(15)
root.right.right = tree_node(7)




# construct binary tree from preorder and inorder lists

def build_tree_pre(preorder, inorder):
	if root == None:
		return root

	inorder_map = {v: i for i, v in enumerate(inorder)}

	def helper(low, high):
		if low > high:
			return 

		root = tree_node(preorder.pop(0))
		mid = inorder_map[root.val]

		root.left = helper(low, mid - 1)
		root.right = helper(mid + 1, high)

		return root

	helper(0, len(inorder) - 1)

	return root

tree_pre = build_tree_pre(preorder, inorder)



def build_tree_post(inorder, postorder):
	if root == None:
		return root

	inorder_map = {v: i for i, v in enumerate(inorder)}

	def helper(low, high):
		if low > high:
			return

		root = tree_node(postorder.pop())
		mid = inorder_map[root.val]

		root.right = helper(mid + 1, high)
		root.left = helper(low, mid - 1)

		return root

	helper(0, len(inorder) - 1)

	return root

tree_post = build_tree_post(inorder, postorder)


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

# print(pre_order(root))
print('pre:', pre_order(tree_pre))


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

print('in:', in_order(root))


def post_order(root):
	output = []
	post_rec(root, output)
	return output

def post_rec(root, output):
	if root:
		post_rec(root.left, output)
		post_rec(root.right, output)
		output.append(root.val)

# print(post_order(root))
print('post:', post_order(tree_post))


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

print('bfs:', bfs(root))


from collections import deque

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

		## optional zig zag
		# if is_level_odd:
		# 	level.reverse()
		# is_level_odd = not is_level_odd

		output.append(level)

	return output

print('lev by lev:', level_by_level(root))


def lev_by_lev_rec(root):
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

print(lev_by_lev_rec(root))




#____________________________________________________________________________#
#____________________________________________________________________________#
#____________________________________________________________________________#

# given a perfectly balanced binary tree, assign each node's next to the node to the right of it, should such a node exist.

class tree_node:
	def __init__(self, x):
		self.val = x
		self.left, self.right, self.next = None, None, None

root = tree_node(1)
root.left = tree_node(2)
root.right = tree_node(3)
root.left.left = tree_node(4)
root.left.right = tree_node(5)
root.right.left = tree_node(6)
root.right.right = tree_node(7)

# recursive
def next_rights(root):
	if root == None:
		return 

	if root.left:
		root.left.next = root.right

	if root.right:
		if root.next:
			root.right.next = root.next.left

	next_rights(root.left)
	next_rights(root.right)

	return root

print(next_rights(root))


# iterative
def next_rights(root):
	if root == None:
		return root

	leftmost = root

	while leftmost.left:
		head = leftmost

		while head:
			head.left.next = head.right

			if head.next:
				head.right.next = head.next.left

			head = head.next

		leftmost = leftmost.left

	return root


print(next_rights(root))





#_______________________________________________________#
#_______________________________________________________#
#_______________________________________________________#

### more practice: ### 
#_______________________________________________________#
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


def post_order(root):	
	if root == None:
		return []

	stack = [root] * 2
	output = []

	while stack:
		current = stack.pop()
		if stack and stack[-1] == current:
			if current.left:
				stack += [current.left] * 2
			if current.right:
				stack += [current.right] * 2
		else:
			output.append(current.val)

	return output

print(post_order(root))



def next_rights(root):
	if root == None:
		return root

	leftmost = root

	while leftmost.left:
		head = leftmost

		while head:
			# if head.left: (optional to declare this if, can just use the logic because .left is already accounted for in the leftmost.left while loop)
			head.left.next = head.right
			if head.next:
				head.right.next = head.next.left

			head = head.next

		leftmost = leftmost.left

	return root

print(next_rights(root))


def next_rights_rec(root):
	if root == None:
		return 

	if root.left:
		root.left.next = root.right

	if root.right:
		if root.next:
			root.right.next = root.next.left

	next_rights_rec(root.left)
	next_rights_rec(root.right)

	return root

print(next_rights_rec(root))


#_______________________________________________________#
#_______________________________________________________#
#_______________________________________________________#