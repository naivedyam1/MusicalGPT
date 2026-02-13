s=isinstance;R=zip;M=sum;L=print;F=len;B=range
import os,math as N,random as S
S.seed(42)
l='input.txt'
if not os.path.exists(l):import urllib.request;urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt',l)
O=[a.strip()for a in open(l).read().strip().split('\n')if a.strip()]
S.shuffle(O);L("num docs:",F(O))
T=sorted(set(''.join(O)));U=F(T);V=U+1;L("vocab size:",V)
class E:
	def __init__(z,d,c=(),g=()):z.data=d;z.grad=0;z._c=c;z._g=g
	def __add__(z,o):o=o if s(o,E)else E(o);return E(z.data+o.data,(z,o),(1,1))
	def __mul__(z,o):o=o if s(o,E)else E(o);return E(z.data*o.data,(z,o),(o.data,z.data))
	def __pow__(z,b):return E(z.data**b,(z,),(b*z.data**(b-1),))
	def log(z):return E(N.log(z.data),(z,),(1/z.data,))
	def exp(z):return E(N.exp(z.data),(z,),(N.exp(z.data),))
	def relu(z):return E(max(0,z.data),(z,),(1.*(z.data>0),))
	__radd__=lambda z,o:z+o
	__rmul__=lambda z,o:z*o
	def __sub__(z,o):return z+o*-1
	def backward(z):
		c=[];d=set()
		def f(v):
			if v not in d:d.add(v);[f(x)for x in v._c];c.append(v)
		f(z);z.grad=1
		for x in reversed(c):
			for p,g in R(x._c,x._g):p.grad+=g*x.grad
n=16;m=4;H=n//m;Y=8
G=lambda o,i,s=.02:[[E(S.gauss(0,s))for _ in B(i)]for _ in B(o)]
C=[G(V,n),G(Y,n),G(V,n),G(n,n),G(n,n),G(n,n),G(n,n,0),G(4*n,n),G(n,4*n,0)]
W=[c for a in C for b in a for c in b];L("num params:",F(W))
I=lambda x,w:[M(a*b for a,b in R(a,x))for a in w]
def Z(l):c=max(a.data for a in l);b=[(a-c).exp()for a in l];d=M(b);return[a*d**-1 for a in b]
def P(x):a=M(v*v for v in x)*F(x)**-1;b=(a+1e-5)**-.5;return[v*b for v in x]
A=lambda a,b:[x+y for x,y in R(a,b)]
def Q(t,p,K,J):
	x=P(A(C[0][t],C[1][p]));g=x;x=P(x);q=I(x,C[3]);k=I(x,C[4]);v=I(x,C[5]);K.append(k);J.append(v);r=[]
	for h in B(m):e=h*H;w=q[e:e+H];ks=[a[e:e+H]for a in K];vs=[a[e:e+H]for a in J];sc=[M(w[i]*ks[j][i]for i in B(H))*H**-.5 for j in B(F(ks))];sp=Z(sc);r.extend(M(sp[i]*vs[i][j]for i in B(F(vs)))for j in B(H))
	x=A(I(r,C[6]),g);g=x;x=A(I([a.relu()**2 for a in I(P(x),C[7])],C[8]),g);return I(x,C[2])
lr,b1,b2,ep=.01,.9,.95,1e-8;d=[.0]*F(W);e=[.0]*F(W);f=500
for p in B(f):
	z=O[p%F(O)];g=[U]+[T.index(a)for a in z]+[U];o=min(Y,F(g)-1);K,J=[],[];q=[]
	for i in B(o):q.append(Z(Q(g[i],i,K,J))[g[i+1]].log()*-1)
	loss=M(q)*o**-1;loss.backward();lr2=lr*.5*(1+N.cos(N.pi*p/f))
	for i,w in enumerate(W):d[i]=b1*d[i]+(1-b1)*w.grad;e[i]=b2*e[i]+(1-b2)*w.grad**2;w.data-=lr2*(d[i]/(1-b1**(p+1)))/((e[i]/(1-b2**(p+1)))**.5+ep);w.grad=0
	L(f"step {p+1:4d} / {f:4d} | loss {loss.data:.4f}")
L('\n--- inference ---')
for j in B(20):
	K,J=[],[];t=U;r=[]
	for p in B(Y):
		k=Z([a*2 for a in Q(t,p,K,J)]);t=S.choices(B(V),[a.data for a in k])[0]
		if t==U:break
		r.append(T[t])
	L(f"sample {j+1:2d}: {''.join(r)}")