"""Monte Carlo check of docs/THEORY_A_PROOFS.md (synthetic data only)."""
import numpy as np
from scipy.stats import norm
rng=np.random.default_rng(0)
d=3; Delta=np.array([.4,-.2,.1]); A=rng.normal(size=(d,d)); Su=.5*A@A.T/d+.1*np.eye(d); Se=np.diag([1,.7,1.3])
Lu=np.linalg.cholesky(Su); Le=np.linalg.cholesky(Se)
def sim_design(a,t,R):
    """a,t: per-site case/control counts. Returns R draws of mean difference."""
    S=len(a); out=np.empty((R,d))
    for r in range(R):
        u=rng.normal(size=(S,d))@Lu.T+.3  # nonzero common mean
        XA=np.concatenate([u[s]+Delta+rng.normal(size=(a[s],d))@Le.T for s in range(S)])
        XT=np.concatenate([u[s]+rng.normal(size=(t[s],d))@Le.T for s in range(S)])
        out[r]=XA.mean(0)-XT.mean(0)
    return out
# Prop 2: k=2, m=20, two-way b=5
k,m,b=2,20,5; a=[m]*k+[b]*k; t=[b]*k+[m]*k
NA,NT=sum(a),sum(t); c=np.array(a)/NA-np.array(t)/NT; L=(c**2).sum()
D=sim_design(a,t,20000)
pred=L*Su+(1/NA+1/NT)*Se
print('Prop2 L',L,'formula',(2/k)*((m-b)/(m+b))**2,'| mean err',np.abs(D.mean(0)-Delta).max(),'| cov maxrel',np.abs(np.cov(D.T)-pred).max()/np.abs(pred).max())
# Cor 2a counterexample k=1,m=2,B=5
def Lval(k,m,u,v):
    NA=k*(m+v);NT=k*(m+u); return k*((m/NA-u/NT)**2+(v/NA-m/NT)**2)
print('Cor2a equal(2.5,2.5)',Lval(1,2,2.5,2.5),'int near-equal(3,2)',Lval(1,2,3,2),'unequal(4,1)',Lval(1,2,4,1))
# Cor 2a: grid check inside budget B<=2km that equal split is optimal (real)
k,m=3,10
worst=0
for B in np.linspace(0,2*k*m,31):
    t_=B/k; us=np.linspace(0,t_,401); Ls=[Lval(k,m,u,t_-u) for u in us]; worst=max(worst,min(Ls)-Lval(k,m,t_/2,t_/2))
print('Cor2a grid: min_L - L_equal (should be >= -1e-15):',worst)
# Prop 3: fixed-effects OLS slope (scalar feature) == weighted within-site mean differences
S=5; a=rng.integers(0,6,S); t=rng.integers(1,6,S); a[0]=0  # site 0 pure TD
x=[];y=[];g=[]
for s in range(S):
    for yy,n in ((1,a[s]),(0,t[s])):
        x+=list(2*s+.7*yy+rng.normal(size=n)); y+=[yy]*n; g+=[s]*n
x,y,g=map(np.array,(x,y,g))
Z=np.column_stack([y]+[(g==s).astype(float) for s in range(S)])
beta=np.linalg.lstsq(Z,x,rcond=None)[0][0]
h=a*t/(a+t); Ds=np.array([x[(g==s)&(y==1)].mean()-x[(g==s)&(y==0)].mean() if a[s]>0 else 0 for s in range(S)])
print('Prop3 FE slope',beta,'weighted',(h*Ds).sum()/h.sum())
# Prop 4: expected reverse gap for plug-in scorer, symmetric two-way design, k=2
k,m,b=2,20,5; a=[m]*k+[b]*k; t=[b]*k+[m]*k; S=2*k
NA,NT=sum(a),sum(t); c=np.array(a)/NA-np.array(t)/NT; L=(c**2).sum()
G=[];Gw=[];Gt=[]
for r in range(40000):
    u=rng.normal(size=(S,d))@Lu.T
    XA=[u[s]+Delta+rng.normal(size=(a[s],d))@Le.T for s in range(S)]
    XT=[u[s]+rng.normal(size=(t[s],d))@Le.T for s in range(S)]
    w=np.concatenate(XA).mean(0)-np.concatenate(XT).mean(0)
    wt=sum((a[s]*t[s]/(a[s]+t[s]))*(XA[s].mean(0)-XT[s].mean(0)) for s in range(S))/sum(a[s]*t[s]/(a[s]+t[s]) for s in range(S))
    xi=u[k]+Delta+rng.normal(size=d)@Le.T   # new case from TD-majority site k
    xj=u[0]+rng.normal(size=d)@Le.T         # new control from ASD-majority site 0
    xt=u[0]+rng.normal(size=d)@Le.T         # new control, same site as a new case from site 0
    xc=u[0]+Delta+rng.normal(size=d)@Le.T
    G.append(w@(xi-xj)); Gw.append(wt@(xi-xj)); Gt.append(w@(xc-xt))
print('Prop4 reverse gap sim',np.mean(G),'formula',Delta@Delta-np.sqrt(2*L/k)*np.trace(Su),'| within-site estimator',np.mean(Gw),'vs',Delta@Delta,'| same-site pair',np.mean(Gt))
# Prop 5: AUC_W and AUC_B for fixed w
w=np.array([1.,-.5,.3]); n=400000
u1,u2=rng.normal(size=(n,d))@Lu.T,rng.normal(size=(n,d))@Lu.T
e1,e2=rng.normal(size=(n,d))@Le.T,rng.normal(size=(n,d))@Le.T
auc_w=np.mean(w@(Delta[:,None]+(e1-e2).T)>0); auc_b=np.mean(w@(Delta[:,None]+(u1-u2+e1-e2).T)>0)
print('Prop5 AUC_W',auc_w,norm.cdf(w@Delta/np.sqrt(2*w@Se@w)),'AUC_B',auc_b,norm.cdf(w@Delta/np.sqrt(2*w@(Se+Su)@w)))
