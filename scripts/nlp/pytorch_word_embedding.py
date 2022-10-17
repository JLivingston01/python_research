
import pandas as pd
import numpy as np


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
#from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

sentences=[
        "what is the weather",
        "the weather outside is frightful",
        "i have a basketball game today",
        "the weather is cold today",
        "the weather is hot today",
        "hot weather is not enjoyable",
        "when is the weather going to get better",
        "the basketball game was long",
        "how much are the basketball tickets",
        "what does the fox say"
        ]

tokens = [i.split() for i in sentences]

all_tokens=[]
for i in tokens:
    all_tokens=all_tokens+i
    
all_tokens=list(set(all_tokens))




all_pairs1=[[[j[i],j[i+1]] for i in range(len(j)-1)] for j in tokens]
all_pairs2=[[[j[i+1],j[i]] for i in range(len(j)-1)] for j in tokens]



token_cooccur=all_pairs1+all_pairs2

token_cooccur[1]

all_pairs=[]
for i in token_cooccur:
    for j in i:
        all_pairs.append(j)
            

X=pd.DataFrame()

X['pairssss']=all_pairs

for i in all_tokens:
    X[i]=X['pairssss'].apply(lambda x: i==x[0])
    
X.drop('pairssss',axis=1,inplace=True)

X=pd.DataFrame(np.where(X,1,0),columns=X.columns)


unique_X=X.drop_duplicates()

Y=pd.DataFrame()

Y['pairssss']=all_pairs

for i in all_tokens:
    Y[i]=Y['pairssss'].apply(lambda x: i==x[1])
    
Y.drop('pairssss',axis=1,inplace=True)

Y=pd.DataFrame(np.where(Y,1,0),columns=Y.columns)




X=np.array(X)
Y=np.array(Y)


X = X.astype(np.float32)
Y = Y.astype(np.float32)


X_torch = torch.from_numpy(X)
Y_torch = torch.from_numpy(Y)

train_ds = TensorDataset(X_torch, Y_torch)



device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

len(all_tokens)

embeddings= 10
configuration = nn.Sequential(
            nn.Linear(len(all_tokens), embeddings),
            nn.Sigmoid(),
            nn.Linear(embeddings, len(all_tokens)),
            nn.Sigmoid()
        )


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stack = configuration

    def forward(self, x):
        logits = self.stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)



loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)



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



train_dataloader = DataLoader(train_ds, batch_size=200)



epochs = 1000000
for t in range(epochs):
    if t%100==0:
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
print("Done!")




weights=list(model.parameters())[0]
bias=list(model.parameters())[1]




embedding=X_torch@weights.T+bias


unique_X.reset_index(drop=True,inplace=True)


unique_X_array=np.array(unique_X).astype(np.float32)
unique_X_torch = torch.from_numpy(unique_X_array)

embedding=unique_X_torch@weights.T+bias


embedding_array=embedding.detach().numpy()
embedding_df=pd.DataFrame(embedding_array)

unique_X_embeddings=unique_X.merge(embedding_df,left_index=True,right_index=True)


cols = [i for i in unique_X_embeddings.columns if i not in range(0,embeddings)]

unique_X_embeddings['argmax']=np.argmax(np.array(unique_X_embeddings[cols]),axis=1)
unique_X_embeddings.sort_values(by='argmax',inplace=True,ascending=True)

unique_X_embeddings['word']=unique_X_embeddings['argmax'].apply(lambda x:
    unique_X_embeddings.columns[x])
    
emb=unique_X_embeddings[list(range(0,embeddings))+['word']]


from sklearn.decomposition import PCA

p=PCA(n_components=2).fit_transform(emb[list(range(0,embeddings))])


fig=plt.figure(figsize=(8,8))
plt.scatter(p[:,0],p[:,1])

for i in range(len(emb)):
    plt.annotate(emb['word'].values[i],(p[i,0],p[i,1]))








