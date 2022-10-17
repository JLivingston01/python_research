


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1/np.float64(1+np.exp(-x))

	

def createweights(s):
	layers=len(s)
	layer=0
	weights=[]
	while layer<layers-1:
		w=np.random.normal(0,.05,(s[layer],s[layer+1]))
		weights.append(w)
		layer=layer+1
	return weights
	
def createbias(s):
	layers=len(s)
	layer=0
	bias=[]
	while layer<layers-1:
		w=np.random.normal(0,.05,(s[layer+1]))
		bias.append(w)
		layer=layer+1
	return bias
	
def predict(train,weights,bias,s):
	layers=len(s)
	layer=0
	predict_on=[train]
	while layer<layers-1:
		pred=sigmoid(predict_on[layer]@weights[layer]+bias[layer])
		predict_on.append(pred)
		layer=layer+1
	return predict_on

def backprop(predict_on,y, weights,bias, s,lr=.01):
	layers=len(s)
	layer=layers-1
	error=predict_on[layer]-y
	while layer>0:
		inn=predict_on[layer-1]
		outt=predict_on[layer]		
		eoo=error*outt*(1-outt)		
		gradw=inn.T@eoo		
		gradb=eoo		
		weights[layer-1]=weights[layer-1]-lr*gradw.reshape(weights[layer-1].shape)
		bias[layer-1]=bias[layer-1]-lr*np.sum(gradb,axis=0)
		error=error@weights[layer-1].T
		layer=layer-1
	return weights,bias


x=np.array([[1,0],[0,1],[1,1],[0,0]])
y=np.array([[1],[1],[0],[0]])

	
s=[2,3,3,1]
weights=createweights(s=s)
bias=createbias(s=s)		



errs = []
for i in range(100000):
    predict_on=predict(x,weights, bias,s=s)
    errs.append(np.sum(abs(predict_on[-1]-y)))
    print(np.sum(abs(predict_on[-1]-y)))
    weights,bias=backprop(predict_on,y, weights, bias, s=s,lr=1)


plt.plot(errs)



#Apply on digits
import pandas as pd 

training_data = pd.read_csv("c:/users/jliv/downloads/mnist_train.csv")
testing_data = pd.read_csv("c:/users/jliv/downloads/mnist_test.csv")


training_labels = training_data['5']
testing_labels = testing_data['7']

training_data.drop(['5'],axis=1,inplace=True)
testing_data.drop(['7'],axis=1,inplace=True)

training_onehot_y = pd.DataFrame()
training_onehot_y['lab'] = training_labels

lr = np.array(list(range(10)))
for i in lr:
    training_onehot_y[i]=np.where(training_onehot_y['lab']==i,1,0)
    
training_onehot_y.drop(['lab'],axis=1,inplace=True)

training_labels.unique()
testing_labels.unique()

testing_map={i:testing_labels.unique()[i] for i in range(len(testing_labels.unique()))}
training_map={i:training_labels.unique()[i] for i in range(len(training_labels.unique()))}

testing_onehot_y = pd.DataFrame()
testing_onehot_y['lab'] = testing_labels

lr = np.array(list(range(10)))

for i in lr:
    testing_onehot_y[i]=np.where(testing_onehot_y['lab']==i,1,0)
    
testing_onehot_y.drop(['lab'],axis=1,inplace=True)

testing_onehot_y=np.array(testing_onehot_y)
training_onehot_y=np.array(training_onehot_y)

   
training_data = np.array(training_data)
testing_data = np.array(testing_data)


    
training_data_flat = training_data.reshape(59999,28*28)
testing_data_flat = testing_data.reshape(9999,28*28)


ex = training_data[1].reshape(28,28)
plt.imshow(ex)

s=[28*28,100,10]
weights=createweights(s=s)
bias=createbias(s=s)		


errs = []
for i in range(30000):
    predict_on=predict(training_data_flat,weights, bias,s=s)
    errs.append(np.sum(abs(predict_on[-1]-training_onehot_y)))
    if i%100 == 0:
        print(i,":",np.sum(abs(predict_on[-1]-training_onehot_y)))
    weights,bias=backprop(predict_on,training_onehot_y, weights, bias, s=s,lr=.00001)
	

    
for i in range(len(weights)):
    np.save("c://users/jliv/documents/weights_layer_"+str(i),weights[i])
for i in range(len(bias)):
    np.save("c://users/jliv/documents/bias_layer_"+str(i),bias[i])
    
train_pred = np.argmax(predict_on[-1],axis=1)
#train_labs = np.argmax(training_onehot_y,axis=1)

#train_pred_mapped=np.array(pd.Series(train_pred).map(training_map))

trainacc = sum(np.where(train_pred==training_labels,1,0))/len(train_pred)


testpredict_on = predict(testing_data_flat,weights,bias,s=s)
test_pred = np.argmax(testpredict_on[-1],axis=1)
#test_labs=np.argmax(testing_onehot_y,axis=1)


#test_pred_mapped=np.array(pd.Series(test_pred).map(testing_map))

testacc = sum(np.where(test_pred==testing_labels,1,0))/len(test_pred)

print(trainacc)
print(testacc)