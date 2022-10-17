
import pandas as pd
import numpy as np

pd.set_option('display.max_columns',500)

X= np.random.normal(0,1,(1000,10))

layers = [5,5,1]

add_bias=True

    
def sigmoid(x):
    
    return 1/(1+np.exp(-x))

def linear(x):
    return x

def d_sigmoid(x):
    
    return sigmoid(x)*(1-sigmoid(x))
def delsig(x):
    #Diagonal is d_sigmoid, but information is lost in that function
    return (sigmoid(x).T@(1-sigmoid(x)))


Y = sigmoid(X@np.random.normal(0,5,10))

#define activation function
act_func = sigmoid


dims = [X.shape[1]]+layers

#init weights
weights = []
for l in range(len(layers)):
    r=dims[l]
    if add_bias:
        r +=1
        
    weights.append(np.random.normal(0,.0005,(r,layers[l])))
    
#forward pass
o=X
if add_bias:
    o = np.hstack([o,np.ones(
        o.shape[0]).reshape(o.shape[0],1)])
outputs = [o]

for i in range(len(weights)):
    
    w=weights[i]
    
    o = outputs[-1]
    
    o = act_func(o@w)
    
    if (add_bias)&(i<len(weights)-1):
        o = np.hstack([o,np.ones(
            o.shape[0]).reshape(o.shape[0],1)])
        
    outputs.append(o)
    
[i.shape for i in outputs]
[i.shape for i in weights]

#back prop


target = Y.reshape(outputs[-1].shape)

l=.001

n_layers = len(layers)

curr_layer = n_layers
last_layer = curr_layer-1

err1 = (target 
        - 
        outputs[-1].reshape(target.shape))

while last_layer>=0:
    
    layer_inputs = outputs[last_layer]
    layer_weights = weights[last_layer]
    
    
    delsig_ = d_sigmoid(layer_inputs@layer_weights)
    #delsig_ = delsig(layer_inputs@layer_weights)
    
    
    err1.shape
    delsig_.shape
    layer_inputs.shape
    layer_weights.shape
    
    grad = (
            layer_inputs.T
            )@(
            err1*(-delsig_).reshape(err1.shape)
            )


    weights[last_layer]=(weights[last_layer]
                         -
                         l*grad.reshape(weights[last_layer].shape))
    
    uw = weights[last_layer][0:weights[last_layer].shape[0]-1]
    
    err1 = err1.reshape(err1.shape[0],
                 uw.T.shape[0])@uw.T

    
    last_layer-=1
    curr_layer-=1
    
    print(grad)

'''
### Main choice
grad = (
        layer_inputs.T
        )@(
        err1*(-delsig_).reshape(err1.shape)
        )
            

### Correct but delsig can be functionalized.. edit.. dims conflicts at bias layers
grad = (
        layer_inputs.T@
        err1@
        (-delsig(layer_inputs@layer_weights)
        )
        )

grad = (
        layer_inputs.T@
        err1@
        (-sigmoid(layer_inputs@layer_weights)).T@
        (1-sigmoid(layer_inputs@layer_weights))
        )
### Naive element multiplication of error and delsig, it isn't the most correct.
grad = (
        layer_inputs.T
        )@(
        err1*
        (-d_sigmoid(layer_inputs@layer_weights)).reshape(err1.shape)
        )
            
'''


class deep_learning:
    
    def __init__(self,lr=.001,layers=[5,5,1],
                 activation_func='Sigmoid',add_bias=True):
        
        self.lr=lr
        self.layers=layers
        self.activation_func=activation_func
        self.act_func=True
         
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def linear(self,x):
        return x
    
    def d_sigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def delsig(self,x):
        #Diagonal is d_sigmoid, but information is lost in that function
        return (self.sigmoid(x).T@(1-self.sigmoid(x)))
    
    def init_weights(self,dims):
        
        self.weights = []
        for l in range(len(self.layers)):
            r=dims[l]
            if add_bias:
                r +=1
                
            self.weights.append(np.random.normal(0,.0005,(r,self.layers[l])))
    
    def forward_pass(self,X):
    
        dims = [X.shape[1]]+self.layers
        
        self.init_weights(self,dims)
        
        if self.activation_func=='Sigmoid':
            self.act_func = self.sigmoid
        elif self.activation_func=='Linear':
            self.act_func = self.linear
      
        #forward pass
        o=X
        if add_bias:
            o = np.hstack([o,np.ones(
                o.shape[0]).reshape(o.shape[0],1)])
        
        outputs = [o]
    
        for i in range(len(weights)):
            
            w=self.weights[i]
            
            o = outputs[-1]
            
            o = act_func(o@w)
            
            if (add_bias)&(i<len(weights)-1):
                o = np.hstack([o,np.ones(
                    o.shape[0]).reshape(o.shape[0],1)])
                
            outputs.append(o)
            
        return outputs
    
    def back_prop(self,outputs,Y):
            
        target = Y.reshape(outputs[-1].shape)
        
        n_layers = len(self.layers)
        
        curr_layer = n_layers
        last_layer = curr_layer-1
        
        err1 = (target 
                - 
                outputs[-1].reshape(target.shape))
        
        while last_layer>=0:
            
            layer_inputs = outputs[last_layer]
            layer_weights = weights[last_layer]
            
            
            #delsig_ = d_sigmoid(layer_inputs@layer_weights)
            delsig_ = delsig(layer_inputs@layer_weights)
            
            
            err1.shape
            delsig_.shape
            layer_inputs.shape
            layer_weights.shape
            
            grad = (
                    layer_inputs.T@
                    err1@
                    (-delsig_
                    )
                    )
        
        
            weights[last_layer]=(weights[last_layer]
                                 -
                                 l*grad.reshape(weights[last_layer].shape))
            
            uw = weights[last_layer][0:weights[last_layer].shape[0]-1]
            
            err1 = err1.reshape(err1.shape[0],
                         uw.T.shape[0])@uw.T
        
            
            last_layer-=1
            curr_layer-=1
            
            print(grad)
        
