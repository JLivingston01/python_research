

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Load Mnist digits
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

training_data[0][0].reshape(
        np.prod(training_data[0][0].shape))

training_data_flat = training_data.reshape(
        (training_data.shape[0],
         training_data.shape[2]*training_data.shape[3]
         ))


#Logistic regression for every pixel in the 28*28 image: 728 targets "Y"
#X is randomly generated. X size determines how many coefficients used here
#More coefficients: more flexibility to generate more variance.

Y=training_data_flat/255


Odds = Y/(1-Y)
Odds = np.where(Odds==0,1e-6,
         np.where(Odds==np.inf,1e6,Odds))

LogOdds=np.log(Odds)

batch_size = 50000
layer_size = 100
rand_vect = np.random.normal(0,1,(batch_size,layer_size))

coefs = (np.linalg.pinv(rand_vect.T@rand_vect)@(
        rand_vect.T@LogOdds[:batch_size]))



pred = 1/(1+np.exp(-(rand_vect@coefs)))
'''
for e in range(0,50):
    plt.imshow(pred[e].reshape((28,28))*255)
    plt.show()
'''

#New Random X can generate new images that never before existed. Some look 
#like digits.
    
new_rand_vect = np.random.normal(0,1,(1,layer_size))

pred = 1/(1+np.exp(-(new_rand_vect@coefs)))
plt.imshow(pred.reshape((28,28))*255)
plt.show()







