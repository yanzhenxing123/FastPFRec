"""
@Time: 2024/7/25 22:59
@Author: yanzx
@Desc: 
"""
import numpy
import numpy as np


with open("./train_origin.txt", "r") as f:
    lines = f.readlines()

user_set = set()
item_set = set()

matrix = [[0 for _ in range(7000)] for _ in range(5000)]

output_lines = []

for line in lines:
    u, i, rating = list(map(int, line.split(" ")))
    if u < 5000 and i < 7000:
        matrix[u][i] = 1
        output_lines.append(line)
    user_set.add(u)
    item_set.add(i)

print(max(user_set))
print(max(item_set))
matrix = np.array(matrix)
print(numpy.sum(matrix) / (5000 * 7000))

s = "".join(output_lines)
with open("./train.txt", "w") as f:
    f.write(s)
