# Assignment – Day 2

## Theory

### 1. What is NumPy?
Answer: NumPy is the liburary of python which is used for apply mathematical operations on a data.

### 2. Why do we use NumPy instead of Python lists?
Anewer: Because it is fast.

### 3. What is an array?
Answer: Array stores data.

### 4. What is vectorization?
Answer: Vectorization is the process in which the operation done in once.

### 5. What is broadcasting?
Answer: Its Like to add a number to all data at once.

### 6. What is the difference between shape and size?
Answer: Shape describe the total number of row and columns, and size tells the total elements in an array.


## Coding

### 1. Create an array of 10 numbers.
arr = np.array([2, 3, 4, 5, 6, 7, 8, 9, 12, 25])

### 2. Multiply every number by 5.
print(arr * 5)

### 3. Add 100 to every number.
print(arr + 100)

### 4. Print shape, size, ndim and dtype.
print("Shape: ", np.shape(arr), "\nSize: ", np.size(arr),"\nDimension: ", np.ndim(arr), "\nData Type: ", arr.dtype)

### 5. Print the first 5 elements.
print(arr[:6])

### 6. Print the last 3 elements.
print(arr[:-4])



- [ ] Completed