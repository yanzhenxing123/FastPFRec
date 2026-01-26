record = []
with open('test_origin.txt') as f:
    for line in f:
        items = line.strip().split()
        for i in items[1:]:
            record.append(items[0]+' '+i+' 1\n')
with open('test_origin_record.txt', 'w') as f:
    f.writelines(record)

record = []
with open('train_origin.txt') as f:
    for line in f:
        items = line.strip().split()
        for i in items[1:]:
            record.append(items[0]+' '+i+' 1\n')
with open('train_origin_record.txt','w') as f:
    f.writelines(record)