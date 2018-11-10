arr = []
for r in range(100):
    row = list(map(int, raw_input().split()))
    arr.append(row)

q = input()
for i in range(q):
    cx, cy, ch = list(map(int, raw_input().split()))
    for x in range(100):
        for y in range(100):
            diff = abs(cx - x) + abs(cy - y)
            if (diff < ch):
                arr[y][x] -= ch - diff

score=0
for r in range(100):
    score += sum(map(abs, arr[r]))
print(score)
