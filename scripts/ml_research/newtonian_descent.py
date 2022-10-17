

import numpy as np

X=np.random.normal(0,1,(100,3))

Y = X@[3,1,-1]



beta = np.random.normal(0,.15,3)

for i in range(1):
    beta = beta - np.linalg.pinv(X.T@X)@(X.T@(X@beta-Y))
    
print(beta)
    

grad = X.T@(X@beta-Y)

hess = X.T@X

beta = beta - np.linalg.pinv(hess)@grad


