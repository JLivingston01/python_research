

import pandas as pd
import numpy as np

pd.set_option('display.max_columns',500)
'''
X= np.random.normal(0,1,(1000,10))

Y = sigmoid(X,np.random.normal(0,5,10))
'''

def sigmoid(x,b):
    #n*1 output
    return 1/(1+np.exp(-(x@b)))
    
class NumpyNetwork:
    
    def __init__(self,lr,layers,add_bias,actfunc='sigmoid',
                 epochs=100,verbose=True):
        
        self.lr=lr
        self.layers=layers
        self.add_bias=add_bias
        self.actfunc=actfunc
        self.epochs=epochs
        self.verbose=verbose
        
        self.derivatives = {self.sigmoid:self.d_sigmoid,
         self.linear:self.d_linear}
        
        
        if self.actfunc=='sigmoid':
            self.act_func = self.sigmoid
        else:
            self.act_func = self.linear
            
        self.grad_fun = self.derivatives[self.act_func]
        
        
        
    def sigmoid(self,x,b):
        #n*1 output
        return 1/(1+np.exp(-(x@b)))
    
    def d_sigmoid(self,err,x,b):
        
        return x.T@(-(self.sigmoid(x,b)*(1-self.sigmoid(x,b)))*err)
    
    def linear(self,x,b):
        #n*1 output
        return x@b
    
    def d_linear(self,err,x,b):
        #k*n output
        return x.T@err
            
    def forward_pass(self,x):
        
        #forward pass
        o=x
        if self.add_bias:
            o = np.hstack([o,np.ones(
                o.shape[0]).reshape(o.shape[0],1)])
        outputs = [o]
        
        for i in range(len(self.weights)):
            
            w=self.weights[i]
            
            o = outputs[-1]
            
            o = self.act_func(o,w)
            
            if (self.add_bias)&(i<len(self.weights)-1):
                o = np.hstack([o,np.ones(
                    o.shape[0]).reshape(o.shape[0],1)])
                
            outputs.append(o)
            
        return outputs
    
    def backwards_pass(self,outputs,y):
        
        #back prop
        
        target = y.reshape(outputs[-1].shape)
        
        n_layers = len(self.layers)
        
        curr_layer = n_layers
        last_layer = curr_layer-1
        
        
        err1 = (target 
                - 
                outputs[-1].reshape(target.shape))
        
        sse = sum(err1**2)
        
        while last_layer>=0:
            
            layer_inputs = outputs[last_layer]
            layer_weights = self.weights[last_layer]
            
            
            #delsig_ = d_sigmoid(layer_inputs@layer_weights)
            grad = self.grad_fun(err1,layer_inputs,layer_weights)
        
            self.weights[last_layer]=(self.weights[last_layer]
                                 -
                                 self.lr*grad.reshape(
                                     self.weights[last_layer].shape))
            
            if self.add_bias:
                dim_drop=1
            else:
                dim_drop=0
            uw = self.weights[last_layer][
                0:self.weights[last_layer].shape[0]-dim_drop]
            
            err1 = err1.reshape(err1.shape[0],
                         uw.T.shape[0])@uw.T
        
            
            last_layer-=1
            curr_layer-=1
            
        if self.verbose:
            print(sse)
        
    def fit(self,x,y):
    
        dims = [x.shape[1]]+self.layers
        
        self.weights = []
        for l in range(len(self.layers)):
            r=dims[l]
            if self.add_bias:
                r +=1
                
            self.weights.append(np.random.normal(0,.0005,(r,self.layers[l])))
            
        
        for e in range(self.epochs):
            
            outputs = self.forward_pass(x)
                
            self.backwards_pass(outputs, y)
            
        return self
            
            
    def predict(self,x):
        
        return(self.forward_pass(x)[-1])
               

'''model = NumpyNetwork(lr=.01, 
                     layers=[5,4,1], 
                     add_bias=True,
                     verbose=True,
                     epochs=1000
                     )

model.fit(X,Y)'''






                 