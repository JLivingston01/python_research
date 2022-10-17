



import pandas as pd 
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
#from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

"""
Loading Mnist Digits locally.

I need to rename the columns clearly for myself, and the labels are integers in
the first column.

I'm going to reshape these flat digits into their 28*28 original written shapes.


"""
training_data = pd.read_csv("c:/users/jliv/downloads/mnist_train.csv")
testing_data = pd.read_csv("c:/users/jliv/downloads/mnist_test.csv")



cols = ['label']+['col_'+str(i) for  i in range(len(training_data.columns)-1)]

training_data.columns = cols
testing_data.columns = cols

training_labels=training_data['label']
testing_labels=testing_data['label']


training_data.drop(['label'],inplace=True,axis=1)
testing_data.drop(['label'],inplace=True,axis=1)

training_data=np.array(training_data).reshape(59999,28,28)
testing_data=np.array(testing_data).reshape(9999,28,28)

training_labels=np.array(training_labels)
testing_labels=np.array(testing_labels)


"""
Pytorch doesn't expect onehot labels. The below code isn't necessary.

n_values = np.max(training_labels) + 1
training_labels_onehot=np.eye(n_values)[training_labels]
n_values = np.max(training_labels) + 1
testing_labels_onehot=np.eye(n_values)[testing_labels]
"""
plt.imshow(training_data[0])
plt.show()


"""
Float tensors should be of numpy type float32 before converting to tensor.

Integer lables do not need to be onehot, but will be cast to LongTensor before 
creating torch dataset.

Numpy arrays will be converted to tensors using from_numpy.

Unsqueezing one dimension of images to explicitely pass images of one channel.

"""
training_data = training_data.astype(np.float32)
testing_data = testing_data.astype(np.float32)
training_labels = training_labels.astype(np.int)
testing_labels = testing_labels.astype(np.int)





training_data_torch = torch.from_numpy(training_data)
testing_data_torch = torch.from_numpy(testing_data)
training_labels = torch.from_numpy(training_labels)
testing_labels = torch.from_numpy(testing_labels)


training_data_torch = training_data_torch.unsqueeze(1) 
testing_data_torch = testing_data_torch.unsqueeze(1) 

training_labels = training_labels.type(torch.LongTensor)
testing_labels = testing_labels.type(torch.LongTensor)

train_ds = TensorDataset(training_data_torch, training_labels)
test_ds = TensorDataset(testing_data_torch, testing_labels)




device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


"""
Configuration using Sequential.

"""
configuration = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Sigmoid()
        )

configuration = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(16),
            nn.Flatten(),
            nn.Linear(16*16, 8*8),
            nn.ReLU(),
            nn.Linear(8*8, 4*4),
            nn.ReLU(),
            nn.Linear(4*4, 10),
            nn.Sigmoid()
        )


configuration = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
           
            nn.AdaptiveAvgPool2d(10),
            nn.Flatten(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.Sigmoid()
        )

"""
Define network class inheriting from nn.Module.

Configuration belongs in the __init__.

forward function returns the output of the network configuration. 

Define model as the created network class and specify device using to()

"""

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten=nn.Flatten()
        self.stack = configuration

    def forward(self, x):
        #flatten = nn.Flatten()
        #x=flatten(x)
        logits = self.stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)



"""
Define loss.

Define optimizer on model parameters and learning rate.
"""


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


"""
Train and test classes are for organization but can be done in loop.

Predict with model(X)
Evaluate loss with loss_fn(pred,y)

set gradient to zero
propogate loss backwards through network
optimize parameters

"""

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



"""
Create a dataloader with model batch size for train and test.

Tell pytorch whether cuda is available or else send processing to CPU
"""
train_dataloader = DataLoader(train_ds, batch_size=1)
test_dataloader = DataLoader(test_ds, batch_size=1)


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")



pred = model(testing_data_torch)
y_test=testing_labels

ypred=pred.argmax(axis=1)

(ypred == y_test).type(torch.float).sum().item()/len(ypred)


