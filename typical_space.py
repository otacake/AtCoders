import sys
input = sys.stdin.readline
N = int(input())
a1 = [int(i) for i in input().split()]
A = int(input())
dp = [[False] * (A+1) for _ in range(N)]
for i in range(N) :
    dp[i][0] = True
if a1[0] <= A:
    dp[0][a1[0]] = True
for i in range(1,N) :
    for a in range(A+1):
        if a-a1[i] <0:
            continue
        if dp[i-1][a-a1[i]]:
            dp[i][a] = True
        if dp[i-1][a]:
            dp[i][a] = True
ans = "NO"
for i in range(N):
    if dp[i][A]:
        ans = "YES"
        break
print(ans)
#部分和
import sys
input = sys.stdin.readline
N,W = [int(i) for i in input().split()]
weight = [0] * N
value= [0] * N
for i in range(N) :
    w,v = [int(i) for i in input().split()]
    value[i] = v
    weight[i] = w
dp = [[0] * (W+1) for _ in range(N+1)]
for i in range(N):
    for w in range(W+1) :
        if w >= weight[i] :
            dp[i+1][w] = max(dp[i][w-weight[i]]+value[i],dp[i][w])
        else :
            dp[i+1][w] = dp[i][w]
print(dp[N][W])
#ナップサック
H,W = [int(i) for i in input().split()]
maze = ['0'] * H
for i in range(H) :
    maze[i] = list(input())
seen = [[False] * 510 for _ in range(510)]
dx = [1,0,-1,0]
dy = [0,1,0,-1]
import sys
sys.setrecursionlimit(10**8)
def dfs(h,w) :
    seen[h][w] = True
    for i in range(4) :
        nh = h + dx[i]
        nw = w + dy[i]
        if nh < 0 or nh >= H or nw < 0 or nw >= W :
            continue
        if maze[nh][nw] == "#" :
            continue
        if seen[nh][nw] :
            continue
        dfs(nh,nw)
for h in range(H) :
    for w in range(W) :
        if maze[h][w] == 's' :
            sh = h
            sw = w
        if maze[h][w] == 'g' :
            gh = h
            gw = w
dfs(sh,sw)
if seen[gh][gw]:
    print("Yes")
else :
    print("No") #typical_dfs_in_maze

import sys
input = sys.stdin.readline
H,N = [int(i) for i in input().split()]
A = [0] * N
B = [0] * N
for i in range(N) :
    a,b = [int(i) for i in input().split()]
    A[i] = a
    B[i] = b
dp = [[(10**9+7)] * (H+1) for _ in range(N)]
for j in range(H+1):
    if j <= A[0]:
        dp[0][j] = B[0]
    if j > A[0]:
        dp[0][j] = dp[0][j-A[0]] + B[0]
for i in range(1,N):
    for j in range(H+1):
        if j == 0:
            dp[i][j] = min(dp[i-1][0],B[i])
        if j <= A[i]:
            dp[i][j] = min(dp[i-1][j],B[i])
        if j > A[i]:
            dp[i][j] = min(dp[i-1][j],dp[i][j-A[i]]+B[i])
print(dp[N-1][H])
#okimochi
R,C = [int(i) for i in input().split()]
sy,sx = [int(i) for i in input().split()]
gy,gx = [int(i) for i in input().split()]
maze = [list(input()) for _ in range(R)]
dx = [1,0,-1,0]
dy = [0,1,0,-1]
import collections
from collections import deque
que = deque()
dic = [[-1] * (C) for _ in range(R)]
dic[sx-1][sy-1] = 0
que.append([sx-1,sy-1])
while (len(que) != 0):
    k = que.popleft()
    x = k[0]
    y = k[1]
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if nx < 0 or nx >= R or ny < 0 or ny >= C:
            continue
        if maze[nx][ny] == "#" :
            continue
        if dic[nx][ny] == -1:
            que.append([nx,ny])
            dic[nx][ny] = dic[x][y] +1
print(dic[gx-1][gy-1])
#bfs_typical
N = input()
n = len(N)
K = int(input())
a = [0] * n
for i in range(n):
    a[i] = int(N[i])
a.insert(0,0)
dp = [[[0,0,0,0],[0,0,0,0]] for _ in range(n+1)]
dp[0][0][0] = 1
for i in range(1,n+1):
    for f in range(2):
        if f == 0:
            for k in range(4):
                if a[i] != 0:
                    if k !=0:
                        dp[i][0][k] = dp[i-1][0][k-1]
                    dp[i][1][k] += dp[i-1][0][k]
                    if k == 3:
                        continue
                    dp[i][1][k+1] += dp[i-1][0][k] * (a[i] - 1)
                if a[i] == 0:
                    dp[i][0][k] = dp[i-1][0][k]
        if f == 1:
            for k in range(4):
                if k!=0:
                    dp[i][1][k] += dp[i-1][1][k] + dp[i-1][1][k-1] * 9
                if k == 0:
                    dp[i][1][k] += dp[i-1][1][k]
ans = dp[n][0][K] + dp[n][1][K]
print(ans)
#keta_dp_typical
from scipy.sparse.csgraph import shortest_path
import numpy as np
H,W = [int(i) for i in input().split()]
c = [0] * 10
for i in range(10):
    c[i] = [int(j) for j in input().split()]
c = np.array(c)
path = shortest_path(c,'D')
A = [0] * H
for i in range(H):
    A[i] = [int(j) for j in input().split()]
ans = 0
for i in range(H):
    for j in range(W):
        if A[i][j] == 1 or A[i][j] == -1:
            continue
        ans += path[A[i][j]][1]
print(int(ans))
#Dijkstra
#Union-Find
class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def all_group_members(self):
        return {r: self.members(r) for r in self.roots()}


#Dijkstra
from heapq import heappush,heappop
INF = 10**18
def dijkstra(s,n,adj): #(始点,node数,隣接リスト)
    dist = [INF]*n
    hq = [(0,s)]
    dist[s] = 0
    seen = [False]*n
    while hq:
        v = heappop(hq)[1]
        seen[v] = True
        for to,cost in adj[v]: #adj[s]はnode sに隣接する(node,重み)をリストで持つ
            if seen[to]== False and dist[v]+cost < dist[to]:
                dist[to] = dist[v]+cost
                heappush(hq,(dist[to],to))
    return dist


#文字列dpの典型
S = input()
n = len(S)
mod = 10**9+7
al = [chr(ord('a') + i) for i in range(26)]

next_index = [[n for _ in range(26)]for _ in range(n+1)]

for i in range(n):
    i = n-1-i
    for j in range(26):
        next_index[i][j] = next_index[i+1][j]
    tmp = ord(S[i]) - ord("A")
    next_index[i][tmp] = i

dp = [0 for _ in range(n+1)]
dp[0] = 1
for i in range(n+1):
    for j in range(26):
        if next_index[i][j] >= n:
            continue
        dp[next_index[i][j]+1] += dp[i]
        dp[next_index[i][j]+1] %= mod

ans = 0
for i in range(1,n+1):
    ans += dp[i]
    ans %= mod
print(ans)
print(dp)

#部分文字列の重複なしの辞書順

S = input()
K = int(input())

T = S[::-1]

n = len(S)
next_index = [[n for _ in range(26)] for _ in range(n+1)]

for i in range(n):
    i = n-1-i
    for j in range(26):
        next_index[i][j] = next_index[i+1][j]
    tmp = ord(T[i]) - ord("a")
    next_index[i][tmp] = i

dp = [[0 for _ in range(26)] for _ in range(n+1)]

for i in range(26):
    dp[0][i] = 1

for i in range(n+1):
    if i == 0:
        for j in range(26):
            if next_index[i][j] >= n:
                continue
            dp[next_index[i][j]+1][j] += 1
    if i != 0:
        tmp = ord(T[i-1])-ord("a")
        for j in range(26):
            if next_index[i][j] >= n:
                continue
            dp[next_index[i][j]+1][j] += dp[i][tmp]



for i in range(1,n):
    for j in range(26):
        dp[i+1][j] += dp[i][j]


dpa = [[0 for _ in range(26)] for _ in range(n+1)]

for i in range(1,n+1):
    for j in range(26):
        dpa[i][j] = dp[n+1-i][j]
print(dpa)

class Bit:
    def __init__(self, n):
        self.size = n
        self.tree = [0]*(n+1)

    def __iter__(self):
        psum = 0
        for i in range(self.size):
            csum = self.sum(i+1)
            yield csum - psum
            psum = csum
        raise StopIteration()

    def __str__(self):  # O(nlogn)
        return str(list(self))

    def sum(self, i):
        # [0, i) の要素の総和を返す
        if not (0 <= i <= self.size): raise ValueError("error!")
        s = 0
        while i>0:
            s += self.tree[i]
            i -= i & -i
        return s

    def add(self, i, x):
        if not (0 <= i < self.size): raise ValueError("error!")
        i += 1
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def __getitem__(self, key):
        if not (0 <= key < self.size): raise IndexError("error!")
        return self.sum(key+1) - self.sum(key)

    def __setitem__(self, key, value):
        # 足し算と引き算にはaddを使うべき
        if not (0 <= key < self.size): raise IndexError("error!")
        self.add(key, value - self[key])