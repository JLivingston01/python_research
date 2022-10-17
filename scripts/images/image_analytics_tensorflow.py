"""

Created on Wed Oct 10 16:31:56 2018



@author: jlivingston

"""



import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from random import shuffle

types = ['airplanes','Bonsai','Faces','Leopards','Motorbikes']

path = 'C://Users/jliv/Downloads/homework4 data/homework data/'

tt = ['Train','Test']

listdir(path+'airplanes'+"/"+"Test"+"/")
#'C://Users/jliv/Downloads/homework4 data/homework data/airplanes/Test/image_0301.jpg'
dat = pd.DataFrame()
for i in range(10*10):
    dat['p'+str(i)] = [0]

dat['y'] = ['init']
#i = 'airplanes'
#j = 'train'
dat = pd.DataFrame()

longy = []
for i in types:
    for j in ['train','test']:
        imlist = listdir(path+i+"/"+j+"/")

        
        for k in imlist:
            im = Image.open(path+i+"/"+j+"/"+k).convert('L').resize((10,10)) # Can be many different formats.
            matrix = np.array(im.getdata())     

            dat = dat.append(pd.DataFrame(matrix).T)
            longy.append(i)

            
dat.reset_index(drop=True,inplace=True)
#labs = np.random.randint(0,9,len(dat))

'''
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10).fit(dat)


clusters = kmeans.predict(dat)
dat['clust'] = clusters
dat['reallab'] = labs
dat['clusterlabs'] = -1

from scipy.stats import mode
for  i  in list(range(0,10)):
    tmp = dat[dat['clust']==i]
    md = mode(tmp['reallab'])[0][0]
    dat['clusterlabs'] = np.where(dat['clust']==i,md,dat['clusterlabs'])
    
acc = sum(np.where(dat['reallab']==dat['clusterlabs'],1,0))/len(dat)

dat = dat.iloc[1:]

import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
cols = list(dat.columns.values)
cols = cols[:len(cols)-1]
'''

#dat.reset_index(inplace = True, drop = True)

#ind = dat.index

#ind = random.sample(ind,len(ind))

#dat = dat.ix[ind]

#

dat['y'] = longy
dat = dat.sample(frac = 1)
dat.reset_index(inplace=True, drop = True)


dat_train = dat[:1800]

dat_test = dat[1800:]

cols = list(dat.columns.values)
cols = cols[:len(cols)-1]

X_train = dat_train[cols]

Y_train = dat_train['y']

X_test = dat_test[cols]

Y_test = dat_test['y']




'''
mod = SVC()
fit = mod.fit(X_train,Y_train)
pred = mod.predict(X_train)

results = pd.DataFrame()
results['Y'] = Y_train
results['Pred'] = pred
results['correct'] = np.where(results['Y'] == results['Pred'],1,0)
len(results['correct'])
np.sum(results['correct'])

predtest = mod.predict(X_test)
testresults = pd.DataFrame()
testresults['Y'] = Y_test
testresults['Pred'] = predtest

testresults['correct'] = np.where(testresults['Y'] == testresults['Pred'],1,0)
len(testresults['correct'])
np.sum(testresults['correct'])







mod = LogisticRegression()
fit = mod.fit(X_train,Y_train)
pred = mod.predict(X_train)

results = pd.DataFrame()
results['Y'] = Y_train
results['Pred'] = pred

results['correct'] = np.where(results['Y'] == results['Pred'],1,0)
len(results['correct'])
np.sum(results['correct'])

predtest = mod.predict(X_test)
testresults = pd.DataFrame()
testresults['Y'] = Y_test
testresults['Pred'] = predtest
testresults['correct'] = np.where(testresults['Y'] == testresults['Pred'],1,0)
len(testresults['correct'])
np.sum(testresults['correct'])'''

import tensorflow
from tensorflow import keras


tf_training = []
tf_training_label = []
for i in range(len(X_train)):
    temp = []
    for j in list(X_train.columns.values):
        temp.append(X_train[j][i])
    tf_training.append(temp)
    tf_training_label.append(Y_train[i])
    

    
tf_training = np.array(tf_training)
tf_training_label = np.array(tf_training_label)


from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

from tensorflow import losses
#tensorflow.nn.sigmoid
model = keras.Sequential([
    keras.layers.Dense(34, activation=tensorflow.nn.sigmoid),
    keras.layers.Dense(68, activation=tensorflow.nn.relu),
    keras.layers.Dense(5,activation = tensorflow.nn.sigmoid)
])


model.compile(optimizer=tensorflow.optimizers.Adam(), 
              #loss=losses.mean_squared_error,
              loss=losses.categorical_crossentropy,
              metrics=['Precision'])

model.fit(tf_training, dummy_y, epochs=500)

model.summary()


'''result = model.predict(tf_training)
dummydf = pd.DataFrame(dummy_y,Y_train)
dummydf.reset_index(inplace = True, drop = False)
dummydfpiv = pd.pivot_table(data = dummydf, index = ['y'], values = [0,1,2,3,4], aggfunc = 'mean')


result_vect = []
result_lab = []
for i in result:
    y = np.argmax(i)
    result_vect.append(y)
    if y == 0:
        result_lab.append('Bonsai')
    if y == 1:
        result_lab.append('Faces')
    if y == 2:
        result_lab.append('Leopards')
    if y == 3:
        result_lab.append('Motorbikes')
    if y == 4:
        result_lab.append('airplanes')
        
prediction = result_lab

results = pd.DataFrame()
results['Y'] = Y_train
results['prediction'] = prediction
results['correct'] = np.where(results['Y']==results['prediction'],1,0)

np.sum(results['correct'])
'''

#Testing

tf_testing = []
tf_testing_label = []
for i in range(min(X_test.index),max(X_test.index)+1):
    temp = []
    for j in list(X_test.columns.values):
        temp.append(X_test[j][i])
    tf_testing.append(temp)
    tf_testing_label.append(Y_test[i])
    

    
tf_testing = np.array(tf_testing)
tf_testing_label = np.array(tf_testing_label)


predresult = model.predict(tf_testing)

encoder2 = LabelEncoder()
encoder2.fit(Y_test)
encoded_Ytest = encoder2.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_ytest = np_utils.to_categorical(encoded_Ytest)

dummydf = pd.DataFrame(dummy_ytest,Y_test)
dummydf.reset_index(inplace = True, drop = False)
dummydfpiv = pd.pivot_table(data = dummydf, index = ['y'], values = [0,1,2,3,4], aggfunc = 'mean')


result_vecttest = []
result_labtest = []
for i in predresult:
    y = np.argmax(i)
    result_vecttest.append(y)
    if y == 0:
        result_labtest.append('Bonsai')
    if y == 1:
        result_labtest.append('Faces')
    if y == 2:
        result_labtest.append('Leopards')
    if y == 3:
        result_labtest.append('Motorbikes')
    if y == 4:
        result_labtest.append('airplanes')

               
prediction = result_labtest
len(Y_test)
len(tf_testing)
len(Y_test)
results = pd.DataFrame()
results['Y'] = Y_test
results['prediction'] = prediction
results['correct'] = np.where(results['Y']==results['prediction'],1,0)

np.mean(results['correct'])

#RGB Analysis




import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from random import shuffle
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import tensorflow

from tensorflow import keras
from tensorflow import losses


types = ['airplanes','Bonsai','Faces','Leopards','Motorbikes']

path = 'C://Users/jliv/Downloads/homework4 data/homework data/'

tt = ['Train','Test']

listdir(path+'airplanes'+"/"+"Test"+"/")
x = 28
y = 28

im = Image.open(path+'airplanes'+"/"+"Test"+"/"+'image_0304.jpg').convert('RGB').resize((x,y)) # Can be many different formats.
matrix = np.array(im.getdata()).reshape(28,28,3)

pics = []
Y = []
for i in types:
    for j in tt:
        imlist = listdir(path+i+"/"+j+"/")
        for k in imlist:
            im = Image.open(path+i+"/"+j+"/"+k).convert('RGB').resize((x,y)) # Can be many different formats.
            matrix = np.array(im.getdata()).reshape(28,28,3)           
            pics.append(matrix)
            Y.append(i)
            
indices = range(0,2361)
training_indices = np.random.choice(indices, size = 1800, replace = False)
testing_indices = [i for i in list(indices) if i not in training_indices]
len(testing_indices)

X_train = [pics[i] for i in training_indices]
Y_train = [Y[i] for i in training_indices]
X_test = [pics[i] for i in testing_indices]
Y_test = [Y[i] for i in testing_indices]

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


encoder2 = LabelEncoder()
encoder2.fit(Y_train)
encoded_Ytrain = encoder2.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_ytrain = np_utils.to_categorical(encoded_Ytrain)




model = keras.Sequential()
model.add(keras.layers.Conv2D(3, kernel_size=(3, 3),activation='linear',input_shape=(28,28,3),padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.1))
model.add(keras.layers.MaxPooling2D((2, 2),padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation=tensorflow.nn.sigmoid))
model.add(keras.layers.Dense(5, activation=tensorflow.nn.sigmoid))

model.summary()

model.compile(optimizer=tensorflow.train.AdamOptimizer(), 
     #         loss=losses.mean_squared_error,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, dummy_ytrain, epochs=600)



predresult = model.predict(X_test)

encoder3 = LabelEncoder()
encoder3.fit(Y_test)
encoded_Ytest = encoder2.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_ytest = np_utils.to_categorical(encoded_Ytest)

dummydf = pd.DataFrame(dummy_ytest,Y_test)
dummydf.reset_index(inplace = True, drop = False)
dummydfpiv = pd.pivot_table(data = dummydf, index = ['index'], values = [0,1,2,3,4], aggfunc = 'mean')


result_vecttest = []
result_labtest = []

for i in predresult:
    y = np.argmax(i)
    result_vecttest.append(y)
    if y == 0:
        result_labtest.append('Bonsai')
    if y == 1:
        result_labtest.append('Faces')
    if y == 2:
        result_labtest.append('Leopards')
    if y == 3:
        result_labtest.append('Motorbikes')
    if y == 4:
        result_labtest.append('airplanes')
        
             
prediction = result_labtest

results = pd.DataFrame()
results['Y'] = Y_test
results['prediction'] = prediction
results['correct'] = np.where(results['Y']==results['prediction'],1,0)

np.sum(results['correct'])

resultpiv = pd.pivot_table(data = results, index=['Y','prediction'], values = ['correct'], aggfunc = 'count' )
model.save("C://Users/Jliv/Downloads/image_classification1.h5")