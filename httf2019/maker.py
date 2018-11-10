import random
arr = [[0 for i in range(100)] for j in range(100)]
for i in range(1000):
    cx = random.randrange(100)
    cy = random.randrange(100)
    ch = random.randrange(100) + 1
    for x in range(100):
        for y in range(100):
            diff = abs(cx - x) + abs(cy - y)
            if (diff < ch):
                arr[y][x] += ch - diff
for y in range(100):
    print(' '.join([str(n) for n in arr[y]]))
