			
			### BINARY SEARCH PRACTICE AND CONCEPT REVIEW ### 


###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###


# find the target value's index in a sorted list using binary search
nums = [10, 20, 30, 40, 50, 60, 70]

def get_target_index(arr, target):
	left, right = 0, len(arr) - 1
	while left <= right:
		pivot = left + (right - left) // 2
		if arr[pivot] == target:
			return pivot
		elif arr[pivot] < target:
			left = pivot + 1
		elif arr[pivot] > target:
			right = pivot -1
	return -1


print(get_target_index(nums, 70))



###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###


# find starting index of a rotated sorted array
rotated_arr_1 = [20, 30, 40, 50, 60, 70, 10]
rotated_arr_2 = [60, 70, 10, 20, 30, 40, 50]

def get_starting_index(arr):
	left, right = 0, len(arr) - 1
	while left < right:
		pivot = left + (right - left) // 2
		if arr[pivot] < arr[right]:
			right = pivot
		elif arr[pivot] > arr[right]:
			left = pivot + 1
	return left

print('Get starting index of rotated sorted array: ', get_starting_index(rotated_arr_1))


# find the ending index of a rotated sorted arrayy
def get_ending_index(arr):
	left, right = 0, len(arr) - 1
	while left < right:
		pivot = left + (right - left) // 2
		if arr[pivot] > arr[right]:
			left = pivot + 1
		elif arr[pivot] < arr[right]:
			right = pivot - 1
	return pivot # <-- or instead: set right = pivot and return left - 1. these two approaches are equivalent.

print('Get ending index of rotated sorted array: ', get_ending_index(rotated_arr_2))



###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###


# find the peak element's value in an unsorted array 
unsorted_nums_1 = [1, 2, 3, 1]
unsorted_nums_2 = [1,2,1,3,5,6,4]

def get_peak_element(arr):
	left, right = 0, len(arr) - 1
	while left < right:
		pivot = left + (right - left) // 2
		if arr[pivot] < arr[pivot + 1]:
			left = pivot + 1
		else:
			right = pivot
	return left

print('Get peak element: ', get_peak_element(unsorted_nums_1))



###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###


# find all numbers in a sorted array less than a target value 
too_many_nums = [2, 4, 5, 6, 7, 8, 10, 12, 15, 17]

def get_less_than_target(nums, target):
	if nums[0] >= target:
		return -1
	if nums[-1] < target:
		return nums

	left, right = 0, len(nums) - 1
	while left < right:
		pivot = left + (right - left) // 2
		if nums[pivot] < target:
			left = pivot + 1
		else:
			right = pivot
	return nums[:left]
	# could return all values above the target by returning nums[left + 1:]

print('Get less than target: ', get_less_than_target(too_many_nums, 10))



###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###


# seeing if left < right construction can find a target value as well
array = [1, 3, 5, 6, 9, 10]

def get_target_index_alternate(arr, target):

	left, right = 0, len(arr) - 1

	while left < right:
		pivot = left + (right - left) // 2
		if arr[pivot] == target:
			return pivot
		if arr[pivot] < target:
			left = pivot + 1
		else:
			right = pivot

	if arr[left] == target:
		return left

	return -1


print('Get target index alternate: ', get_target_index_alternate(array, 13))


###_________________________________________________________________________###
###_________________________________________________________________________###
###_________________________________________________________________________###



'''
Some thoughts on all the above: 


when to use left < right and right = pivot vs left <= right and right = pivot - 1:
left < right and right = pivot:
	when finding the index of a value or the value itself in a rotated sorted array (to find the lowest value, return left. to find the highest value, return left -1.)
	can you use this to find the index of a value in a non-rotated sorted array? 
		yes! you simply compare arr[pivot] to target, and outside of the while loop check if arr[left] == target, then return left. otherwise, return -1.
	finding the first bad version (which is basically a rotated sorted array.)
		anytime you're trying to find the beginning of something.
	generally use this version. you don't always need a pivot either:
		just know that when left = right, you will exit out of the loop and you can then return the essentially failed circumstance. if you get what you're looking for before left = right, then return it still in the while loop.
	finding a peak element (note that the array does not need to be sorted here! neat.)
		because of this, instead of comparing arr[pivot] to right, compare it to arr[pivot] to arr[pivot + 1]. 

left <= right and right = pivot -1:
	when finding the index of a target value in an array.
'''


'''
    IN CONCLUSION
_____________________

1. Always default to using left < right and right = pivot. 
2. Be sure to think about how the actual specifics of the problem may require you to compare arr[pivot] against either a target value or against arr[right]. 	
	2a. You'll always return some variation of left outside of the while loop if you're not looking for a target.
	2b. If you are looking for a target, you'll return pivot in the loop and outside, run a check for the last element and return left if so, otherwise the target is not in the array.

'''

