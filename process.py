with open("../false.txt", 'r') as f:
    lines = f.readlines()
false = []
for line in lines:
    line = line.rstrip()
    false.append(line)
with open("labeled-data-id", 'r') as f:
    lines = f.readlines()
label = []
for line in lines:
    line = line.rstrip()
    label.append(line)
for i in label:
    if i in false:
        print(i)
        label.remove(i)
with open("label_data_id_new",'w') as f:
    f.write('\n'.join(label))
