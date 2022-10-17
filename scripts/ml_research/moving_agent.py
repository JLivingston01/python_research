

import numpy as np

import matplotlib.pyplot as plt


class agent:
    
    """
    Agent has X and Y position, and fuel energy.
    Fuel enables the agent to move up to manhattan(1,1) to find more fuel.
    """
    def __init__(self,x,y,fuel,eats):
        self.x=x
        self.y=y
        self.fuel=fuel
        self.reach=.75
        self.eats=eats
        
    def get_posit(self):
        return(np.array([self.x,self.y]))
        
    def set_posit(self,x,y):
        self.x=x
        self.y=y
        
    def get_fuel(self):
        return(self.fuel)
        
    def set_fuel(self,fuel):
        self.fuel=fuel
        
    def breathe(self,space):
        d=np.sum(abs(space[:,1:]-agent1.get_posit()),axis=1)
        spacetemp=np.column_stack((space,d))
        
        spacetemp=spacetemp[spacetemp[:,3].argsort()]
        
        if len(spacetemp[(spacetemp[:,0]==self.eats)&
                 (spacetemp[:,3]<=self.reach)][:,0])>=1:
            
            
            found=len(spacetemp[(spacetemp[:,0]==self.eats)&
                     (spacetemp[:,3]<=self.reach)][:,0])
            
            self.fuel=found
            
            spacetemp[:,0]=np.where((spacetemp[:,0]==self.eats)&
                     (spacetemp[:,3]<=self.reach),1-self.eats,spacetemp[:,0])
            
            self.move()
        
        else:
            print("No Fuel in range")
        
        return spacetemp[:,0:3]
    
    def move(self):
        if self.fuel >0:
            self.fuel -= 1
            
            posit=self.get_posit()
            delta=np.random.uniform(-1,1,2)
            
            self.set_posit(posit[0]+delta[0],posit[1]+delta[1])
        else:
            print("No Fuel: Can't Move")
            

a=np.random.binomial(1,.5,size=100)
b=np.random.normal(5,2.5,size=(100,2))
space=np.column_stack((a,b))

plt.scatter(space[:,1],space[:,2],color=np.where(space[:,0]==1,'red','blue'))
plt.show()


init_posit1=np.random.normal(5,1.5,size=2)
agent1=agent(init_posit1[0],init_posit1[1],2,1)


init_posit2=np.random.normal(5,1.5,size=2)
agent2=agent(init_posit2[0],init_posit2[1],2,0)


space=agent1.breathe(space)
agent1.get_fuel()

space=agent2.breathe(space)
agent2.get_fuel()


