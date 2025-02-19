import numpy as np

# 1. Генерація масивів
fixed_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
random_array = np.random.random((4, 3))
range_array = np.arange(1, 16, 2)
ones_array = np.ones(9, dtype=int)
zeros_array = np.zeros((2, 4), dtype=float)
linspace_array = np.linspace(0, 1, 5)
normal_array = np.random.normal(1, 10, size=(2, 5))
randint_array = np.random.randint(0, 100, (4, 4))
empty_array = np.empty(5)
new_array = fixed_array[0:3:2, -2:]

# 2. Індексація та підмасиви
first_element = fixed_array[0, 0]
last_element = fixed_array[-1, -1]
subarray_2d = fixed_array[0:2, 1:3]
subarray_1d = fixed_array[0, :]
subarray_1d_from_1d = range_array[2:6]
reversed_array = fixed_array[::-1, ::-1]

# 3. Арифметичні операції
add_result = fixed_array + 2
multiply_result = fixed_array * 3
divide_result = fixed_array / 2
power_result = fixed_array ** 2
mod_result = fixed_array % 2
negate_result = -fixed_array

# 4. Операції reduce, accumulate, outer
array_to_reduce = np.arange(0, 6)
reduce_result = np.add.reduce(array_to_reduce)
accumulate_result = np.add.accumulate(array_to_reduce)
outer_result = np.multiply.outer(np.arange(1, 10), np.arange(1, 10))

# 5. Статистичні характеристики
sum_value = np.sum(fixed_array)
min_value = np.min(fixed_array)
max_value = np.max(fixed_array)
mean_value = np.mean(fixed_array)
std_value = np.std(fixed_array)
var_value = np.var(fixed_array)
median_value = np.median(fixed_array)
percentile_50 = np.percentile(fixed_array, 50)


def print_results():
    print("Fixed Array:\n", fixed_array)
    print("Random Array:\n", random_array)
    print("Range Array:\n", range_array)
    print("Ones Array:\n", ones_array)
    print("Zeros Array:\n", zeros_array)
    print("Linspace Array:\n", linspace_array)
    print("Normal Distribution Array:\n", normal_array)
    print("Random Integer Array:\n", randint_array)
    print("Empty Array:\n", empty_array)
    print("First Element:\n", first_element)
    print("Last Element:\n", last_element)
    print("Subarray 2D:\n", subarray_2d)
    print("Subarray 1D:\n", subarray_1d)
    print("1D Subarray from 1D Array:\n", subarray_1d_from_1d)
    print("Reversed Array:\n", reversed_array)
    print("Addition Result:\n", add_result)
    print("Multiplication Result:\n", multiply_result)
    print("Division Result:\n", divide_result)
    print("Power Result:\n", power_result)
    print("Modulo Result:\n", mod_result)
    print("Negation Result:\n", negate_result)
    print("Array to Reduce:\n", array_to_reduce)
    print("Reduce Result:\n", reduce_result)
    print("Accumulate Result:\n", accumulate_result)
    print("Outer Result:\n", outer_result)
    print("Sum:\n", sum_value)
    print("Min:\n", min_value)
    print("Max:\n", max_value)
    print("Mean:\n", mean_value)
    print("Standard Deviation:\n", std_value)
    print("Variance:\n", var_value)
    print("Median:\n", median_value)
    print("50th Percentile:\n", percentile_50)
    print("New Array:\n", new_array)


print_results()
