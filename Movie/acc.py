with open("r1.txt", r) as f:
    lines = f.readlines()
r1=[]
for line in lines:
    line = line.rstrip()
    r1.append(line)
with open("1000.txt", r) as f:
    lines = f.readlines()
gt_video = []
num_right = 0
num_wrong = 0
num = 0
for line in lines:
    line = line.rstrip()
    line = line.split(' ')
    gt = line[1]
    if gt == r1:
        num_right = num_right + 1
    else:
        num_wrong = num_wrong +1
    num = num + 1
print(num_right, num_wrong)