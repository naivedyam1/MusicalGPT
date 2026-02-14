s=int;L=len;B=range;P=print
import wave as W,struct as S,numpy as N
from collections import Counter as C
d=open("primeGPT_decimal.txt").read().strip();P("digits:",L(d))
q=C(d);r=[x for x,_ in q.most_common()]
T={-1:"'Ni",0:"Sa",2:"Re",4:"Ga",5:"Ma",7:"Pa",8:"Komal Dha",11:"Ni",12:"Sa'",16:"Ga'"}
M={};M[r[0]]="Sa";M[r[1]]="Pa";M[r[2]]="Sa'"
M[r[-1]]="'Ni";M[r[-2]]="Ga'"
for i,v in enumerate(["Re","Ga","Ma","Komal Dha","Ni"]):M[r[3+i]]=v
V={v:k for k,v in T.items()}
[P(f"{x}({q[x]})={M[x]}[{V[M[x]]:+d}]") for x in r]
w=W.open("harmonium.wav","rb");R=w.getframerate();n=w.getnframes()
a=N.array(S.unpack(f"<{n}h",w.readframes(n)),dtype=N.float64)/32768.0;w.close()
b=a[s(.05*R):s(.05*R)+s(.5*R)];O=s(.25*R);F=s(.005*R)
def Z(x,t,o,f):
	r=2.**(t/12.);k=s(L(x)/r)
	if k<2:k=2
	g=N.interp(N.linspace(0,L(x)-1,k),N.arange(L(x)),x)
	g=g[:o]if L(g)>=o else N.tile(g,(o//L(g))+1)[:o]
	if f>0 and f<o//2:g[:f]*=N.linspace(0,1,f);g[-f:]*=N.linspace(1,0,f)
	return g
E={x:Z(b,V[M[x]],O,F)for x in M}
P("assembling...");o=N.concatenate([E[c]for c in d])
p=N.max(N.abs(o))
if p>0:o=o/p*.95
o=(o*32767).astype(N.int16)
f=W.open("musicalGPT.wav","wb");f.setnchannels(1);f.setsampwidth(2);f.setframerate(R);f.writeframes(o.tobytes());f.close()
P(f"wrote musicalGPT.wav | {L(o)/R:.0f}s | {L(d)} notes | 🎵")
