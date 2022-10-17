
import pandas as pd
import numpy as np
from PIL import Image
from os import listdir
from sklearn.cluster import KMeans
from scipy.stats import mode

path = 'C://Users/jliv/Downloads/homework4 data/homework data/'


    
dat = pd.DataFrame()

for i in ['airplanes']:
    for j in ['train']:
        imlist = listdir(path+i+"/"+j+"/")

        
        for k in imlist:
            im = Image.open(path+i+"/"+j+"/"+k).convert('L').resize((28,28)) # Can be many different formats.
            matrix = np.array(im.getdata())     

            dat = dat.append(pd.DataFrame(matrix).T)
            
import matplotlib.pyplot as plt
plt.imshow(im)
            
path = "c:/users/jliv/downloads/IMG_7559.jpg"
im = Image.open(path).convert('L').resize((28,28)) # Can be many different formats.
im=im.rotate(angle=270)
matrix = np.array(im.getdata())     
np.set_printoptions(threshold=np.inf)
matrix.reshape((28,28))
matrix1 = np.where(matrix>100,255,0).reshape((28,28))

plt.imshow(matrix1)

dat.reset_index(drop=True,inplace=True)
labs = np.random.randint(0,9,len(dat))



kmeans = KMeans(n_clusters=10).fit(dat)


clusters = kmeans.predict(dat)
dat['clust'] = clusters
dat['reallab'] = labs
dat['clusterlabs'] = -1


for  i  in list(range(0,10)):
    tmp = dat[dat['clust']==i]
    md = mode(tmp['reallab'])[0][0]
    dat['clusterlabs'] = np.where(dat['clust']==i,md,dat['clusterlabs'])
    
acc = sum(np.where(dat['reallab']==dat['clusterlabs'],1,0))/len(dat)




