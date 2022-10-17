
import numpy as np

def iq_mod(X,B):
    return 1/(1+np.exp(-X@B))

def delsig(X,B):
    return iq_mod(X,B)*(1-iq_mod(X,B))

X = np.random.normal(0,5,(100,6))
B = np.random.normal(0,1,6)

iq_mod(X,B)


s=np.array(list(range(1,100)))
X = s+np.random.normal(0,5,99)

X

plt.plot(X)

Xd=pd.DataFrame()
Xd['x']=X

Xd

Xd['x1']=Xd['x'].shift(1)
Xd['x2']=Xd['x'].shift(2)
Xd['x3']=Xd['x'].shift(3)
Xd['x4']=Xd['x'].shift(4)
Xd['x5']=Xd['x'].shift(5)
Xd['x6']=Xd['x'].shift(6)
Xd['x7']=Xd['x'].shift(7)

Xd.dropna(inplace=True)

Xd

from sklearn.decomposition import PCA

pca1 = PCA(n_components=7)

a = pca1.fit_transform(Xd) 




plt.plot(a.T[1])

pd.DataFrame(a).corr()


#GET UCI TIME SERIES DATA AND TRY THIS OUT