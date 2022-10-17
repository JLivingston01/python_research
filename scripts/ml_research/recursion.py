
import numpy as np    
import time


a = np.random.normal(5,5,(6,6))

def determinant(a):
    opp = a[0]
    l=len(opp)
    rr = list(range(l))
    if l==1:
        return opp
    else:
        
        trow = opp*(-1)**(np.array(rr))
        
        slices = [[i for i in rr if i != j] for j in rr]
        
        submats = np.array([a[1:,i] for i in slices])
        
        res = sum(np.array([determinant(submats[i])*trow[i] for i in rr]))
        
        return res

t0 = time.time()  
determinant(a)[0]
t1=time.time()
print(t1-t0)


t0 = time.time()  
np.linalg.det(a)
t1=time.time()
print(t1-t0)


def fact(x):
    if x==0:
        return 1
    else:
        return x*fact(x-1)
    
    
fact(6)