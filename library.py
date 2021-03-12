


k = int(input("input your number:")) #furui
for n in range (2,k+1) :
    for x in range (2,n) :
        if n % x == 0 :
            print (n,"is not a prime number")
            break
    else :
        print (n, "is a prime number")

def fib(N) : #fibon
    list = []
    a,b = 0,1
    list.append(a)
    list.append(b)
    for i in range(2,N) :
        c = list[i-1] + list[i-2]
        list.append(c)
    return list[N-1]

N = int(input("number:"))
print(fib(N))

def func(li,b) : #nibutankazari
    sum = 0
    for i in range(len(li)) :
        sum+= abs(li[i] - b - i - 1)
    return sum
def okimati(m) :
    if m == 0 :
        return m
    else :
        return 1

def binary_search(li) : #nibutanashu
    left = -1000000001
    right = 1000000001
    while right - left > 1 :
        med = (right + left) // 2
        x = func(li,med)
        mdx = func(li,med-1)
        pdx = func(li,med+1)
        if mdx > x and pdx < x :
            left = med
            continue
        elif mdx < x and pdx > x :
            right = med
            continue
        elif mdx >= x and pdx >= x :
            return x
    else :
        return -1

def wa(N) : #kakuketanowa
    S = str(N)
    array = list(map(int,S))
    return sum(array)

S = list(input())
import collections
count = 0
for i in range(N) : #mojiretunoshurui,ooikedo
    a = 0
    Sa = S[:i]
    Sb = S[i:]
    ca = collections.Counter(Sa)
    cb = collections.Counter(Sb)
    ka = list(ca.keys())
    kb = list(cb.keys())
    for j in ka :
        for k in kb :
            if j == k :
                a +=1
    count = max(a,count)

import sys
input = sys.stdin.readline #yomikomikousokuka

import fractions
from fractions import gcd
def lcm(x,y) :
    return(x*y) // fractions.gcd(x,y)

#junretsurekkyo(tuple de detekurkara chuui)
li = [1,2,3,4,5,6]
import itertools
from itertools import permutations
c = list(itertools.permutations(li))

#mojiretsu ga suuji ka handann(bool ninaru)
s = "333"
if s.isdecimal() :
    print(45)

# list wo mojiretsu ni
li = ["34","rtry"]
a = ''.join(li)

s = input() #alphabet hitotsu sakino moji
print(chr(ord(s) + 1))#ord(a)=97

al = [chr(ord('a') + i) for i in range(26)] #alphabet ni benri

import sys
input = sys.stdin.readline
N = int(input())
a = [int(i) for i in input().split()]
aa = a[:]
aaa = a[:]
for i in range(N) :
    aa[i] -=1
for i in range(N) :
    aaa[i] += 1
A = a + aa + aaa
import collections
c = collections.Counter(A)
li = list(c.values())
li.sort()
print(li[-1]) #kakusuuji no shutsugen kaisuu uresii

import collections
from collections import deque
s = deque()
#que no dashikata

stra = "hinano"
K = stra.upper() #oompji henkan
k = stra.lower() #komoji

dp = [[0] * 10 for _ in range(10)] #dp[10][10]

H,W = [int(i) for i in input().split()]
maze = ['0'] * H
for i in range(H) :
    maze[i] = list(input())
seen = [[False] * 510 for _ in range(510)]
dx = [1,0,-1,0]
dy = [0,1,0,-1]
import sys
sys.setrecursionlimit(100000000)
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

import math
print(math.factorial(int(input()))%int(1e9+7))
#kaijou_raku

import math
def comb(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
#combination
mod = 10 ** 9 + 7

mod = 10**9+7
def frac(N):
    ans = 1
    ans1 = 1
    ans2 = 1
    ans3 = 1
    for i in range(2,N+1):
        if i % 3 == 0:
            ans1 *=i
            ans1 %= mod
        if i % 3 == 1:
            ans2 *=i
            ans2 %= mod
        if i % 3 == 2:
            ans3 *=i
            ans3 %= mod
    ans = ans1*ans2*ans3 % mod
    return ans
#frac_rapid
import numpy as np
dic= [[0,1,2],[0,9,9],[0,0,0]]
arr = np.array(dic)
M = np.amax(arr)
#matrix_maximum
#graph_no_okimochi
Graph = [[] for _ in range(10**6)]
#comb_rapid
p = 10 ** 9 + 7
N = 10 ** 6
fact = [1, 1]
factinv = [1, 1]
inv = [0, 1]
def cmb(n, r, mod):
    if (r < 0) or (n < r):
        return 0
    r = min(r, n - r)
    return fact[n] * factinv[r] * factinv[n-r] % p
for i in range(2, N + 1):
    fact.append((fact[-1] * i) % p)
    inv.append((-inv[p % i] * (p // i)) % p)
    factinv.append((factinv[-1] * inv[-1]) % p)
#int_float_judge
a = 123.045
if a.is_integer():
    print("YES")


#soinsuu_bunkai
def factorization(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n**0.5//1))+1):
        if temp%i==0:
            cnt=0
            while temp%i==0:
                cnt+=1
                temp //= i
            arr.append([i, cnt])

    if temp!=1:
        arr.append([temp, 1])

    if arr==[]:
        arr.append([n, 1])

    return arr
#eg_factorization(24)
#[[2,3],[3,1]]
import heapq
N,M = [int(i) for i in input().split()]
A = [int(i) for i in input().split()]
for i in range(len(A)):
    A[i] = A[i]*(-1)
heapq.heapify(A)
for i in range(M):
    maxa = heapq.heappop(A)
    maxa = -1 * maxa
    maxa = maxa//2
    maxa = -1 * maxa
    heapq.heappush(A,maxa)
print(-sum(A))
#how_to_use_heapq
def divisore(n):
    divisors=[]
    for i in range(1,int(n**0.5)+1):
        if n%i==0:
            divisors.append(i)
            if i!=n//i:
                divisors.append(n//i)
    divisors.sort(reverse=True)
    return divisors
#yakusuu_rekkyo
ca,sa,fa = list(map(int,input().split()))
#izatoiu
#エラトステネスのふるい
def get_sieve_of_eratosthenes(n):
    prime = [2]
    limit = int(n**0.5)
    data = [i + 1 for i in range(2, n, 2)]
    while True:
        p = data[0]
        if limit <= p:
            return prime + data
        prime.append(p)
        data = [e for e in data if e % p != 0]
#furui_rekkyo
import bisect
#index-nibutan

#reverse_matrix_O(n^3)??
N = int(input())
A = [int(i) for i in input().split()]
for i in range(N):
    A[i] *= 2
import numpy as np
mat = [[0] * N for _ in range(N)]
for i in range(N):
    j = i + 1
    if j == N:
        j = 0
    mat[i][i] = 1
    mat[i][j] = 1
AB = np.array(mat)
inv_A = np.linalg.inv(AB)
ans = list(np.dot(inv_A,A))
for i in range(N):
    ans[i] = int(ans[i])
print(*ans)
#ruijou_rapid
import sys
sys.setrecursionlimit(100000000)
def pow_k(a,n,mod):
    if n == 0:
        return 1
    if n % 2 ==0:
        return pow_k(a*a % mod,n//2,mod)
    else:
        return a * pow_k(a,n-1,mod) % mod
def modinv(a, mod):
    return pow(a, mod-2, mod)
def combination_list(n, mod):
    lst = [1]
    for i in range(1, n+1):
        lst.append(lst[-1] * (n+1-i) % mod * modinv(i, mod) % mod)
    return lst
def combination(n, r, mod=10**9+7):
    r = min(r, n-r)
    res = 1
    for i in range(r):
        res = res * (n - i) * modinv(i+1, mod) % mod
    return res
def chofuku_cmb(n, r, mod=10**9+7):
    return combination(n+r-1, r, mod)
#combination_rapid
def base10to(n, b):
    if (int(n/b)):
        return base10to(int(n/b), b) + str(n%b)
    return str(n%b)
#10->n(sinhou)
def len_check(k,n):
    if len(k) == n:
        return k
    else:
        for _ in range(n-len(k)):
            k = '0'+k
        return k
#myself-bit全探索の長さ揃える
from itertools import product
#直積
from sys import setrecursionlimit
setrecursionlimit(10**6)
#template
from collections import Counter
def inputlist(): return [int(j) for j in input().split()]
#template

from itertools import combinations_with_replacement
#chohuku_rekkyo_tuple

#how_to_make_class(object)

class Hinatazaka_member():
    name = ''
    height = 0
    cute = 0
    def __init__(self,h,c,n): #インスタンスが作られる時に実行できる関数
        self.height = h
        self.cute = c
        self.name = n
    def return_height(self):
        return self.height
    def return_cute(self):
        return self.cute
    def return_name(self):
        return self.name

def osa_k_prepro(N):
    sieve = [i for i in range(N+1)]
    p = 2
    while p*p <= N:
        if sieve[p] == p:
            for i in range(2*p,N+1,p):
                if sieve[i] == i:
                    sieve[i] = p
        p +=1
    return sieve

sieve = osa_k_prepro(N)

def osk_k(A):
    lis = []
    while A> 1:
        lis.append(sieve[A])
        A = A // sieve[A]
    return lis

#巡回セールスマン

def tsp(d):
    n = len(d)
    # DP[A] = {v: value}
    DP = dict()

    for A in range(1, 1 << n):
        if A & 1 << 0 == 0:# 集合Aが0を含まない
            continue
        if A not in DP:
            DP[A] = dict()

        # main
        for v in range(n):
            if A & 1 << v == 0:
                if A == 1 << 0:
                    DP[A][v] = d[0][v] if d[0][v] > 0 else float('inf')
                else:
                    DP[A][v] = min([DP[A ^ 1 << u][u] + d[u][v] for u in range(n)
                                    if u != 0 and A & 1 << u != 0 and d[u][v] > 0]
                                  + [float('inf')])
    # 最後だけ例外処理
    V = 1 << n
    DP[V] = dict()
    DP[V][0] = min([DP[A ^ 1 << u][u] + d[u][0] for u in range(n)
                    if u != 0 and A & 1 << u != 0 and d[u][0] > 0]
                    + [float('inf')])


    return DP[V][0]


#Decimalによる10進法の計算
s = str(3.1415926)
import decimal
x = decimal.Decimal(s)
y = 100

z = x*y
za = z.quantize(decimal.Decimal('0.01'),rounding=decimal.ROUND_HALF_UP)
#↑任意の小数桁による四捨五入

class Combination:
    """
    O(n)の前計算を1回行うことで，O(1)でnCr mod mを求められる
    n_max = 10**6のとき前処理は約950ms (PyPyなら約340ms, 10**7で約1800ms)
    使用例：
    comb = Combination(1000000)
    print(comb(5, 3))  # 10
    """
    def __init__(self, n_max, mod=10**9+7):
        self.mod = mod
        self.modinv = self.make_modinv_list(n_max)
        self.fac, self.facinv = self.make_factorial_list(n_max)

    def __call__(self, n, r):
        return self.fac[n] * self.facinv[r] % self.mod * self.facinv[n-r] % self.mod

    def make_factorial_list(self, n):
        # 階乗のリストと階乗のmod逆元のリストを返す O(n)
        # self.make_modinv_list()が先に実行されている必要がある
        fac = [1]
        facinv = [1]
        for i in range(1, n+1):
            fac.append(fac[i-1] * i % self.mod)
            facinv.append(facinv[i-1] * self.modinv[i] % self.mod)
        return fac, facinv

    def make_modinv_list(self, n):
        # 0からnまでのmod逆元のリストを返す O(n)
        modinv = [0] * (n+1)
        modinv[1] = 1
        for i in range(2, n+1):
            modinv[i] = self.mod - self.mod//i * modinv[self.mod%i] % self.mod
        return modinv

#組み合わせのmod

def segfunc(x, y):
    return min(x, y)

#セグメント木

class SegTree:
    """
    init(init_val, ide_ele): 配列init_valで初期化 O(N)
    update(k, x): k番目の値をxに更新 O(N)
    query(l, r): 区間[l, r)をsegfuncしたものを返す O(logN)
    """
    def __init__(self, init_val, segfunc, ide_ele):
        """
        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        n: 要素数
        num: n以上の最小の2のべき乗
        tree: セグメント木(1-index)
        """
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.tree = [ide_ele] * 2 * self.num
        # 配列の値を葉にセット
        for i in range(n):
            self.tree[self.num + i] = init_val[i]
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.segfunc(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, k, x):
        """
        k番目の値をxに更新
        k: index(0-index)
        x: update value
        """
        k += self.num
        self.tree[k] = x
        while k > 1:
            self.tree[k >> 1] = self.segfunc(self.tree[k], self.tree[k ^ 1])
            k >>= 1

    def query(self, l, r):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        res = self.ide_ele

        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segfunc(res, self.tree[l])
                l += 1
            if r & 1:
                res = self.segfunc(res, self.tree[r - 1])
            l >>= 1
            r >>= 1
        return res

import os, sys, base64, zlib
code = b'c%0Q*TW{OQ6@KThm{qU>Y0Fx#(Wl6E;W%>EsN<-X5?~ty14<l8On6&zlt^{l-`;c149Ss5O0Klsc7enu=klHFoXgPG3r|e_NU*`Ld^o)QN1u=Q^<Dq!-g@CjlVITqc0O3ic^q};m*rS1lj&k2LlKTe+MTzyuRi}OCNfShEf|^tJ;wQ)`}fzJ8%xAe{*;Wlm;K~sSv=vBIFc?f2Ixt0S-ViCLhvLG{K-l!T4a7SW04z*EO94-Wzy?a205Qa3!aR<tY-m{Ou{l(`V;0ZWXvXUnDAs3k3X{u_VJTFSiKv62JVlTTH9v9JnOLPa5~SNM=KR_kw(lO$8q4~i}-^-nO_62n~GO3VG~5EaQNNG=`rgmXB0%sgaqO!-MLi|>R8bt2)JA&qGNet65x|vvg>rLbj(R5(w%iIdWus|q@3i-K*Wqku_yQ<@mwja%bxW@1eq{&3zQqQ`2V7*8b!g+^HNQ$PB$cK<oQ#paQCZ+VV_?;ycycK7zlK`U5EKo`-j2mDxL9xn`UCPNCIKI=?v!eGR{9-(LT-{hiNJ9-{0L?>PWDmkblY9d9*Z_ne6GumoI%OQWuT`^I_0jZ9D8GtXwlHEV47%9gCSC*+B5|X?=@2S*84!Z3aUc2%hqt9USh04)F0mA)u^g4z(F>R21N;o5>m!YZ|9DE|JhS2d#N~fBn$Eu@khZ^{;h3JY)OozOfyhh`F9;Y*_+`+J@c=UWc9{yMC03RN70}=A-kZ-Kn3Vb-?mzHxsbMgPkRRxKdjg>??p8o|(dbr!Kr93s2=y1+NI>uVRB4A_p9R`+17~)W$IyB*PT$G+6c1l!HpNKHz;h;eQi29H$+C>#!2tLGSBXrB04~nTv<eshbH~4Q-*w@8d`mq8_1dnj&SwO(%0c_h)m6DB`z;8#Krl#NS}WI9@zM%pF8Sbf!l{fW4XjPhmS37pIQ3sivV!AtGos-A0~RYDKjGflbpmWNySnn8+1Ii&mZba`v?gZ(+(^{;^CYsrYIa%4=pUi4%Uf#x%;lS~Cwp26Mz^jVo$~U{Q?_2WwcP9HQm@1VL`OWPt!>=%`hMib9xmi7_4U2&x(_^RB}JOOB8CJyArAMBN}(MD8Y}laoy>8S?<j3PaFx4ktiWgVh~8@O$yCG$n?FNHQO$PkY<rycvfRvI)Tr`y+lk>c6#-3Z$On;|^1KKr@Tu>%7i{$>#wfNm+h}rRw~%jNUE6V6bWHk@|h&HV6(j4|N(3;%L@pzK8J|pYahYX!LJ*VwYeN_E&m*Nk5O7jmsRvb&(gn36^h6v+v&v$aA)NWN+d$bfs-o4y@IhBng&Dnp@`M9E%r{#Z#8L(M&KKDO~*Mb&vr3c3PeCMA4vCr_IfV!_2~byiyf%E|HU0#FemyZc1KzaoS`4xnAEx--Du29v>G4=9cLECouwW`e!!4et}yV0;OInYl@JuDMlI7RgV=<`4_RGf=5X}fQp0D+S>nYL&9%!KS0*b*bn7FXYIC8tz%_(hUFiMe1>5U2%zYuELsgu*eT^b$#s%IJSqb;^e_+nm;Tl3wOap(+7#Rd*+`JbjG*aJlNtjD2$vecud(mx3GN9u2;zyYUa?{frr_b82!s^$c5;dM5VA(U4*E9DpIpj>PoY({4acDw%BG4+$)6?_K|V2gJEG>y>@`)(R=$7v=3&!m^47EPCm12;*g+6tNC~1aPfJ(XVA0?ZJyY}MN6-qHbe_SMMw!gr6vr@qef$KRJ=!$nw1P~+X~P5cWL72tC0U&|YC6}e$b+==4Lmf{O)L_G4D|h7tD^~ZSQ}u<jB@fbVG*>#Mh3O@fEwtYtv73AQoK4+CRGlvyChT$L<oY^+537fCSPnt6Q<Aru*%W1Zy>i57B1*nh{9+u3Dmq<P1o(-nw8Oq^0YjQB}nH>XmC{h62Uo@h>LkzX}QvFV_S`pu^OUKYzj}0xa*_;sjP=eDIm}rf~%s2Y8vdXO{vVOo>3MlXmwh)7zytLs&*pF6jjsdwj+nl^)|0|%<69mq{{o17CKuSBO|rxDkkR4o?Ifb?h|2(di#<|Be=o%tA!tUAKYLe6evQW0{I9!b5a<;!G3&e@X*cq!NP3jnNlK6p%m)kg3b}E#*w0RFGW5%*&*^)>FWCQ|DvdM<cu4WJp{lG_Sp<?vJ<Yg?)y-O)^mUA*tZ@+J@4T6=AgI&IMnm7l3=?yJw#VUpZ~ja$}`=BYgVxr->6AS3zd-D>y731oW505qrpGOIk0O#j0x5FAHqc5^Y6_^acNMwIccDXo%`O@sO>*QUezz`O<&&@{15WLBu-Yl_%5cY7b(hMqsVTTMSrEWs|=8s&?d^WU2?*brb6W?SDsm}$TNqEr?ul+Q<TfI9iBv2A)8t$0)}eIw|Wx+sD2w)9~kU+87RLUSWn9Vre#HaYkalUnm^q2R2O}4(P1>uMi?XPX2l0p{*2yCe^bsHnOmC4q2ZI{uHpy!bMmVG-y68VD)>1W3GV^6*5s*3%*#^^l@G8ZP{bmDj^W77)Qzb6jc2_$?79oDgl61HXYPIEPw=DhO&W*#?x>|U+dn?w9gqT5$Mpd?x)UxGzxt4tAEk<AW!MK2C|)VwTOsGM_gYMga*PHt)eKvYE$9_m6nCy~7STldag;|ZnCY;${n0OYwSk@l4c(RJ;e~hDdg{PlRh2hvSCbWl2O2g&YE0gMShKPMKcqDVJk)Lta4=KZ{zgdrMCON!(8O?;O&IR79>ZO3VCXus0Y}ePum*8b!;h}?wqe!qMOYQSu!?E`!l_#6aDsis@SypDBB+TOn5xCV2|r$BCXPNX4N!Wvg7((^BGl~yeVf8hQiKM%r2z?MYI&lIayfVDNna{vEQuk~;Gajh8kkCu256eAfcNS1#53Ws&S*f#sS5Zt46aE{#%+`wXPc!Tuhf-o;b_fHR_kD;V=0*W-85Ivgt{f4mOykxM}N&z1hpZYeGyHB3T|0(p^`y}P%{+VZ0J*_Ksl!huN`Kc>i;&?D3CkNC53*9WwbONe};ca8c&>!yl1R3IlN!sUvX9Wi)fs<JKZ2s_uuxT-xhwla*RAzQfs(NCiKDTgPZDfi?`DJ_WpKct0ME%$v;eV)h@6SeP)fPI*B<seNU+C?l^TeE~RU@HF7AtwKt_+AImMZLj!mdHMqIEdNYKqcXRdrZp6MbWAN40u#b1&J7Ct6-)^n@(lj={uLD7m{S5ONo?&F<*joewGqSQlNblNc#bjx;;ww_CW0C!=lb<^T3CrKT#wGmb{rxpS-vXapEw=GHf@7IgVW_W+(MO=HaJ`k;j5zdR%lzvrpb%2MqL|{>sc@yR+a}wp7!Hej?T!8ArK(9uH8lk;)D*f?2{D@G9%<Cmtjg!3Q3tJKm*Zhs8YzCDFw*ZKzWf_ya-ux'
code_setup = '\nfrom distutils.core import setup, Extension\nmodule = Extension(\n    "cppset",\n    sources=["set_wrapper.cpp"],\n    extra_compile_args=["-O3", "-march=native", "-std=c++14"]\n)\nsetup(\n    name="SetMethod",\n    version="0.2.0",\n    description="wrapper for C++ set",\n    ext_modules=[module]\n)\n'
if sys.argv[-1] == "ONLINE_JUDGE" or os.getcwd() != "/imojudge/sandbox":
    with open("set_wrapper.cpp", "w") as f:
        f.write(zlib.decompress(base64.b85decode(code)).decode("utf-8"))
    with open("setup.py", "w") as f:
        f.write(code_setup)
    os.system(f"{sys.executable} setup.py build_ext --inplace")

from cppset import CppSet

from math import gcd

class LDE:
    #初期化
    def __init__(self,a,b,c):
        self.a,self.b,self.c=a,b,c
        self.m,self.x0,self.y0=0,[0],[0]
        #解が存在するか
        self.check=True
        g=gcd(self.a,self.b)
        if c%g!=0:
            self.check=False
        else:
            #ax+by=gの特殊解を求める
            self.extgcd(self.a,self.b,self.x0,self.y0)
            #ax+by=cの特殊解を求める
            self.x0=self.x0[0]*c//g
            self.y0=self.y0[0]*c//g
            #一般解を求めるために
            self.a//=g
            self.b//=g

    #拡張ユークリッドの互除法
    #返り値:aとbの最大公約数
    def extgcd(self,a,b,x,y):
        if b==0:
            x[0],y[0]=1,0
            return a
        d=self.extgcd(b,a%b,y,x)
        y[0]-=(a//b)*x[0]
        return d
    def returnAnswer(self):
        xy = [self.x0,self.y0]
        return xy

import math
def comb(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))