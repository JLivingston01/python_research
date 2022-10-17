
import pandas as pd


import matplotlib.pyplot as plt

import numpy as np


#https://datahub.io/core/s-and-p-500
dat = pd.read_csv("c:/users/jliv/downloads/data_csv.csv")

dat['year']=dat['Date'].apply(lambda x: x[:4])
dat['year']=dat['year'].astype(int)

i = dat.index.values/max(dat.index)

print("Has the SP500 grown strictly exponentially? It can be shown that an exponential model a second exponential model on the exponent can well describe the historical growth of the sp500.")
plt.plot(dat['SP500'])
plt.title("monthly s&p 500: big exponential growth")
plt.xlabel("month since 1871")
plt.show()

print("If growth is exponential, evaluating log(sp500) can render growth linear. It is observable that the log-trend still grows more in recent history than in early history.")
plt.plot(np.log(dat['SP500']))
plt.title("monthly log s&p 500: growing faster and faster, but linear in part")
plt.xlabel("month since 1871")
plt.show()

print("Using a homomorphism on i (normalized month since 1871), functionally a**i for some a, 'a' can be chosen to map the normalized month to a new value in space a**i, such that small i are mapped to a closer proximity to each other than large i.")
a=6
plt.scatter(i,a**i)
plt.title("i (month since 1871, normalized), maps to a**i \n stretching high i and compressing low i")
plt.xlabel("i = month since 1871 normalized")
plt.ylabel("a**i")
plt.show()

print("Initial guess of a is "+str(a)+", such that plot log_a(sp500) against a**i is as linear as possible.")
plt.plot(a**i,np.log(dat['SP500'])/np.log(a))
plt.title("warping x axis: a**x; a="+str(a)+", compressing low x \n stretching high x, even more linear")
plt.xlabel("a**i")
plt.ylabel("log_a(sp500)")
plt.show()


print("With initial a = "+str(a)+", fitting linear model X=[a**i,1] to log_a(sp500) for inital guess of beta.  log_a(sp500)=b1*i'+b0 where i'=a**i")
X = np.column_stack([a**i,np.ones(len(i))])
Y=np.log(dat['SP500'])/np.log(a)
beta = np.linalg.inv(X.T@X)@(X.T@Y)

Yhat=X@beta

plt.plot(a**i,Y)
plt.plot(a**i,Yhat)
plt.title("Fitting a linear model this can be abstracted into an exponential curve \n with respect to a power-warped axis")
plt.xlabel("a**i")
plt.ylabel("log_a(sp500)")
plt.show()


print("Plotting log_a(Y) over unmorphed i scale, exponential growth is observable. Here log_5(sp500)=b1*a**i+b0")
plt.plot(i,Y)
plt.plot(i,Yhat)
plt.ylabel("i")
plt.title("This growth is well behaved with respect to a**i, and thus, i")
plt.show()

print("Plotting Y over unmorphed i scale, growth is exponential, and the exponent grows exponentially. Here sp500=a**(b1*a**i+b0). The shape is largely correct, but parameters a, b1 and b0 can be tuned with gradient descent.")
plt.plot(i,a**Y)
plt.plot(i,a**Yhat)
plt.ylabel("i")
plt.title("This growth is well behaved with respect to a**i, and thus, i")
plt.show()


print("Setting Y to be sp500, x is i (normalized month since 1870).")
print("Model is Y=a**(b1*a**i+b0), from visual analysis above, initial a guessed "+str(a)+", and model fit given a="+str(a)+" yielded beta vector "+ str(beta)+".")

print("Derivative of SSE cost function is evaluated using each of a, b1 and b0.")
print("In terms of a: (a**(b1*a**X+b0)-Y)@(a**(b1*a**X+b0-1)*(b1*a**X+b0+b1*X*np.log(a)*a**X))")
print("In terms of b1: (a**(b1*a**X+b0)-Y)@(np.log(a)*a**(b1*a**X+b0+X))")
print("In terms of b0: (a**(b1*a**X+b0)-Y)@(np.log(a)*a**(b1*a**X+b0))\n")

Y=dat['SP500']
X = i

l=.00000000001
tol=1e-8
a,b1,b0=a,beta[0],beta[1]
e1=1e5
for c in range(15000):
    err = (a**(b1*a**X+b0)-Y)
    
    e2=np.mean(abs(err))
    
    if e1-e2<tol:
        break
    else:
        e1=e2
    grad_mod_a = a**(b1*a**X+b0-1)*(b1*a**X+b0+b1*X*np.log(a)*a**X)
    grad_mod_b1=np.log(a)*a**(b1*a**X+b0+X)
    grad_mod_b0 = np.log(a)*a**(b1*a**X+b0)
    
    grad_a=err@grad_mod_a
    grad_b1=err@grad_mod_b1
    grad_b0=err@grad_mod_b0
    
    a,b1,b0=a-l*grad_a,b1-l*grad_b1,b0-l*grad_b0
    

print(np.mean(abs(err)))
print("After gradient descent, new a, b1 and b0 are:",a,b1,b0)
yhat=a**(b1*a**X+b0)

print("With learned a, b1 and b0, model well-defines historical growth of the market.")
plt.plot(Y)
plt.plot(yhat)
plt.show()

