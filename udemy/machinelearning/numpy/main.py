import numpy as np

# 1D array
arr = np.array([n for n in range(0, 5)])
print('{}\n'.format(arr))

# 2D array
mat = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
print('{}\n'.format(mat))

# np.arange(start,stop,step)
print(np.arange(0, 10, 2))
print(np.arange(20).reshape(4, 5))

# ones
print(np.ones((3, 2)))

# zeros
print(np.zeros((2, 3)))

# linspace
print(np.linspace(0, 5, 5))

# eye
print(np.eye(3))

# random uniform distribution [0,1]
print(np.random.rand(2, 2))

# random discrete uniform distribution [start,end]
print(np.random.randint(1, 100, 3))

# random standard normal distribution
print(np.random.randn(3))

# reshape function
print(np.array([n for n in range(0, 10)]).reshape(2, 5))

# min, max
print('max: {}, min: {}'.format(arr.max(), arr.min()))
print('max position: {}, min position: {}'.format(arr.argmax(), arr.argmin()))

# shape
print(arr.shape)

# dtype
print(arr.dtype)

# np.array indexing
arr = np.arange(0, 11)
print('{} {} {}'.format(arr[1:5], arr[:7], arr[2:]))

arr[0:3] = 0
print(arr)

# slices reference
arr_slice = arr[0:6]
arr_slice[:] = 1
print('arr: {}, slice: {}'.format(arr, arr_slice))

# copy
copy_arr = arr.copy()
copy_arr[:] = 0
print('arr: {}, copy: {}'.format(arr, copy_arr))

# matrix
matrix = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print('double brackets: {} or: {}'.format(matrix[0][0], matrix[0, 1]))

# sub-matrix
print('{} \n{}'.format(matrix[:3, :2], matrix[:2]))

# conditional selection
arr_bool = arr > 5
print(arr[arr_bool])
print(arr[arr % 2 == 0])

# array operations
print(arr + arr)
print(arr - arr)
print(arr * arr)
print(arr + 100)

print(np.sqrt(arr))
print(np.exp(arr))
print(np.sin(arr))

# runtime warning: -inf
print(np.log(arr))





