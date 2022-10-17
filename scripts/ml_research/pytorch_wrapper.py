
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader

class NeuralNetwork(nn.Module):
    
    def __init__(self,
                 configuration,
                 loss_fn,
                 optimizer,
                 lr,
                 batch_size=200,
                 epochs=500):
        super(NeuralNetwork,self).__init__()
        self.stack=configuration
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.lr=lr
        self.batch_size=batch_size
        self.epochs=epochs
        
        self.set_optimizer()
    
    def set_optimizer(self):
        
        optimizer = self.optimizer(self.parameters(),lr=self.lr)
        self.optimizer=optimizer
    
    def forward(self,x):
        logits=self.stack(x)
        return logits
    
    def fit_one(self,X,Y):
        
        X_torch=torch.from_numpy(np.array(X).astype(np.float32))
        
        Y_torch=torch.from_numpy(np.array(Y).astype(np.float32))
    
        if type(self.loss_fn).__name__!='MSELoss':
            Y_torch = Y_torch.type(torch.LongTensor)

    
        train_ds=TensorDataset(X_torch,Y_torch)
        
        train_dataloader = DataLoader(train_ds,self.batch_size)
        
        losses = []
        
        for batch,(xt,yt) in enumerate(train_dataloader):
            
            X1,y1=xt.to('cpu'),yt.to('cpu')
            
            pred = self(X1)
            
            if type(self.loss_fn).__name__!='CrossEntropyLoss':
                loss = self.loss_fn(pred.reshape(y1.shape),y1)
            else:
                loss = self.loss_fn(pred,y1)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
        print('loss:', np.mean(losses))
        
    def fit(self,X,Y):
        
        for i in range(self.epochs):
            
            self.fit_one(X,Y)
            
            if i%100==0:
                print(f"Epoch {i+1}\n--------")
                
    def predict(self,X):
        X_torch=torch.from_numpy(np.array(X).astype(np.float32))
        ds = TensorDataset(X_torch)
        
        dataloader = DataLoader(ds,batch_size = len(ds))
        
        with torch.no_grad():
            
            for xt in dataloader:
                X1=xt[0].to('cpu')
                
                pred = self(X1)
                
        return pred
    
    
    
X = np.random.normal(5,2,(10000,30))

Y = X@np.random.normal(1,2,30)


model = NeuralNetwork(
        configuration= nn.Sequential(
                nn.Linear(30,10),
                nn.Sigmoid(),
                nn.Linear(10,1)
                ),
        loss_fn = torch.nn.modules.loss.L1Loss(),
     #   loss_fn=torch.nn.modules.loss.MSELoss(),

        optimizer = torch.optim.SGD,
        lr = 1e-2,
        batch_size = 200,
        epochs=100
        ).to('cpu')
        
model.fit(X,Y)
        

pred = model.predict(X)
        
Y_torch = torch.from_numpy(Y.astype(np.float32))
Y_torch = Y_torch.reshape(pred.shape)


1-sum((Y_torch-pred)**2)/sum((Y_torch-torch.mean(Y_torch))**2)



###
training_data = pd.read_csv("c:/users/jliv/downloads/mnist_train.csv")
testing_data = pd.read_csv("c:/users/jliv/downloads/mnist_test.csv")



cols = ['label']+['col_'+str(i) for  i in range(len(training_data.columns)-1)]

training_data.columns = cols
testing_data.columns = cols




training_labels=training_data['label']
testing_labels=testing_data['label']


training_data.drop(['label'],inplace=True,axis=1)
testing_data.drop(['label'],inplace=True,axis=1)

training_data=np.array(training_data).reshape(59999,1,28,28)
testing_data=np.array(testing_data).reshape(9999,1,28,28)

import matplotlib.pyplot as plt
plt.imshow(training_data[0][0])
plt.show()

training_labels=np.array(training_labels)
testing_labels=np.array(testing_labels)




model = NeuralNetwork(
        configuration= nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Flatten(),
            nn.Linear(16*16, 10),
            nn.Sigmoid()
                ),
        loss_fn = torch.nn.modules.loss.CrossEntropyLoss(),
        optimizer = torch.optim.SGD,
        lr = 1e-2,
        batch_size = 200,
        epochs=1
        ).to('cpu')
        
model.fit(training_data,training_labels)

pred=np.argmax(model.predict(training_data),axis=1)


Y_torch = torch.from_numpy(training_labels.astype(np.float32))
Y_torch = Y_torch.reshape(pred.shape)


np.mean(np.where(Y_torch==pred,1,0))


pred=np.argmax(model.predict(testing_data),axis=1)

Y_torch = torch.from_numpy(testing_labels.astype(np.float32))
Y_torch = Y_torch.reshape(pred.shape)


np.mean(np.where(Y_torch==pred,1,0))

