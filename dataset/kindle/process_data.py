"""
@Time: 2024/7/25 22:59
@Author: yanzx
@Desc: 
"""
import numpy
import numpy as np
with open("./valid_origin.txt", "r") as f:
    lines = f.readlines()

user_set = set()
item_set = set()

matrix = [[0 for _ in range(9173)] for _ in range(7650)]


output_lines = []

for line in lines:
    u, i, rating = list(map(int, line.split(" ")))
    if u < 7650 and i < 9173:
        matrix[u][i] = 1
        output_lines.append(line)
    user_set.add(u)
    item_set.add(i)

matrix = np.array(matrix)
print(numpy.sum(matrix) / (9173 * 7650))
#
# s = "".join(output_lines)
# with open("./valid.txt", "w") as f:
#     f.write(s)

