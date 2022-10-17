
"""
A model for large positive outcomes can be developed using logistic regression
with no activation function. Consider a logistic regression:
    1. X@B = ln(Y/(1-Y)) -> Y = 1/(1-e**(-X@B))
Y is a probability between 0 and 1, and Y/(1-Y) represents the "odds" of each
observation. ln(Y/(1-Y)) then represents the "log odds", and the linear model
X@B is fit to the log odds of observations. Odds are necessarily greater than 
zero, with no upper bound, thus when exponentiating the log odds linear model,
we obtain a model for a potentially very large outcome:
    2. e**(X@B) = Y/(1-Y) = L
Therefore, given a large positive outcome and predictors, an exponential model
akin to (2) can be fitted using the same techniques used to fit a logistic 
regression. This is done by calculating a value between 0 and 1 for each
outcome in the same way that one would calculate a probability given odds:
    3. Y = L/(1+L)
Logistic regression can then be fit to predict the left side of (3) for each 
observation. Predictions for L can then be expressed as:
    4. predL = sigmoid(X@B)/(1-sigmoid(X@B))
Further, predicted numeric outcomes can be expressed as the product of model
factors corresponding to each predictor in X:
    e**(X@B) = L = e**(x1*b1)*e**(x2*b2)*...*e**(xn*bn)
    

"""


import numpy as np
import matplotlib.pyplot as plt


#FAKE RANDOM DATA AND 
x = np.random.normal(10,3,(100,2))

beta = [1.5,.25]

L = np.exp(x@beta+np.random.normal(0,1,100))

y=L/(1+L)



def sigmoid(x):
    return 1/(1+np.e**(-x))

def del_sig(x):
    return sigmoid(x)*(1-sigmoid(x))

def predict(X,B):
    return sigmoid(X@B)


coefs = np.random.normal(0,.05,2)

lr = 1e0
for i in range(200000):
    
    grad = x.T@((predict(x,coefs)-y)*del_sig(predict(x,coefs)))
    
    coefs=coefs-lr*grad
    
    if i%1000 ==0:
        print(grad)
    
predsigy = predict(x,coefs)

predsales=predsigy/(1-predsigy)
    
plt.scatter(L,predsales)
plt.xlim(0,1e10)
plt.ylim(0,1e10)
plt.show() 

print(np.mean(abs(np.log(predsales)-np.log(L))))
print(np.mean(abs(predsales-L)))

#exp model alternative
logL = np.log(L)
coefs2 = np.linalg.inv(x.T@x)@(x.T@logL)
pred = np.exp(x@coefs2)


print(np.mean(abs(np.log(pred)-np.log(L))))
print(np.mean(abs(pred-L)))



###

plt.hist(sigmoid(x@[-3,3]))

y = np.where(sigmoid(x@[-3,3])>.5,1,0)

L = y/(1-y)
L = np.where(L==np.inf,1e13,1e-13)
logL = np.log(L)

coefs2 = np.linalg.inv(x.T@x)@(x.T@logL)

predL = np.exp(x@coefs2)


predynum=predL/(1+predL)

predy = np.where(predynum>.5,1,0)

np.where(y == predy,1,0)




