#template
def inputlist(): return [int(j) for j in input().split()]
#template

N = int(input())

ans = 0

for i in range(N):
    if i == 0:
        continue
    else:
        ans += N/i

print(ans)
