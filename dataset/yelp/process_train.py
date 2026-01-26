"""
@Time: 2024/5/13 19:16
@Author: yanzx
@Desc: 
"""

record = []
with open('train_origin.txt') as f:
    for line in f:
        items = line.strip().split()
        user_id = items[0]
        item_ids = items[1:]
        weight = 1
        for item_id in item_ids:
            record.append(user_id + ' ' + item_id + ' 1\n')

print(record)

with open('train.txt', 'w') as f:
    f.writelines(record)
