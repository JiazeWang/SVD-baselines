with open("shot_list.txt", 'r') as f:
    lines = f.readlines()
shot = []
for line in lines:
    line = line.rstrip()
    shot.append(line)

with open("train_test_groundtruth", 'r') as f:
    lines = f.readlines()
all = []
for line in lines:
    line = line.rstrip()
    id = line.split(' ')[0]
    if id in shot:
        all.append(line)
    else:
        continue
with open("useful_225_train_test_groundtruth.txt", 'w') as f:
    f.write("\n".join(all))
