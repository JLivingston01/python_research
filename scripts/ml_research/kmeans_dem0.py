

import numpy as np

d1 = np.random.normal(3,.5,(10,3))
d2=np.random.normal(5,.5,(8,3))

d3=np.random.normal(7,.5,(8,3))


d = np.vstack((d1,d2,d3))

centroids=3

c=np.random.normal(np.mean(d),np.std(d),(centroids,d.shape[1]))


def kmeans(dat,centroids,max_iter):
    
    d = dat    
    c=np.random.normal(np.mean(d),np.std(d),(centroids,d.shape[1]))
    
    def mydist(d,c):
        distarray=[]
        for i in range(c.shape[0]):
            distarray.append(np.sum((d-c[i])**2,axis=1)**.5)
        
        distarray=np.array(distarray)
        return distarray   
        
    for j in range(16):
        dists = mydist(d,c).T
        clusts=np.argmin(dists,axis=1)
        for i in range(centroids):
            c[i]=np.mean(d[clusts==i],axis=0)
        
    return clusts
        
kmeans(d,3,16)