import torch
import random
import numpy as np
q=0
#结果产生器
def resultgenerator(x):
    
        return [[ 2*i*i -3*i+3 for i in eachx ] for eachx in x]

class fullconnection:
    def __init__(self,inputnum,outputnum):
        self.w=[[(random.random()-0.5)*2 for i in range(inputnum)] for i in range(outputnum)]
        self.b=[0 for i in range(outputnum)] 
        self.lr=0.0002
    def forward(self,x):
        if q==1:print("x",x)        
        if q==1:print("w",self.w)
        if q==1:print('b',self.b)
        self.x=x
        output=[]
        for eachx in x:
            outtemp=[]
            for i,m in zip(self.w,self.b):
                temp=0
                for j,k in zip(i,eachx):
                    temp=temp+j*k
                temp=temp+m
                outtemp.append(temp)
            output.append(outtemp)

            
        self.output=output
        if q==1:print("output",output,'\n')
        return output
    def backward(self,dy):
        dx=[]
        for eachdy,eachx in zip(dy,self.x):
            dxtemp=[]
            for i in range(len(eachx)):
                temp=0 
                for j,k in  zip([m[i] for m in self.w],eachdy):
                    temp=temp+j*k
                dxtemp.append(temp)
            dx.append(dxtemp)
 
    
        for n in range(len(self.w)):      
            for k in range(len(self.w[0])):
                self.w[n][k]=self.w[n][k]-self.lr* sum([eachdy[n]*eachx[k]  for eachdy,eachx in zip(dy,self.x)] )
        
        for i in range(len(self.b)):
            self.b[i]=self.b[i]-self.lr*sum([m[i] for m in dy])
 
        return dx        
    def __call__(self,x):
        return self.forward(x)
    def __str__(self):
        return "fullconnection input:"+str(len(self.w[0]))+"output:"+str(len(self.w))
        #return "w:"+str(self.w)+"b:"+str(self.b)
class relu:
    def __init__(self):
        pass
 
    def forward(self,x):
        self.x=x
        return [[i if i>-0 else 0.1*i for i in eachx ]for eachx in x]
        
    def __call__(self,x):
        return self.forward(x)
    def backward(self,dy):
        return [[ j if i > -0 else 0.1*j for i,j in zip(eachx,eachdy)]for eachdy,eachx in zip(dy,self.x)]
        
    def __str__(self):
        return "relu"
#函数模拟器
class model:
    def __init__(self,*num):
        lastnum=num[0]
        self.layer=[]
        for i in  num[1:]:
 
            self.layer.append(fullconnection(lastnum,i))
            if i!=num[-1]:
                self.layer.append(relu())
            
            lastnum=i
 
    def forward(self,x):
        result=x
        for i in self.layer:
           result=i.forward(result)
        return result
        
    def backward(self,loss):
        dy=loss
        for i in reversed(self.layer):
           dy=i.backward(dy)
 
    def __call__(self,x):
        return self.forward(x)
    
m=model(1, 200,20,2,1)
for i in range(100000):
    if q==1:print('----------------------')
    batch=160
    x= [[(random.random()-0.5)]for i in range(batch)]#(random.random()-0.5) *4
    #x=[[0.1],[-0.3],[0.3] ,[-0.3],[0.3]]
    y=m.forward(x)
    t=resultgenerator(x)
    #print(t)
    loss=[]
    for eachy,eacht in zip(y,t):
        loss.append([i-j for i ,j in zip(eachy,eacht)])
 

    #print("all:",x,y[0],t,loss)
   
 
    m.backward(loss)
    if i%5==0:
        z1=m.forward([[0.1]])
        q1=resultgenerator([[0.1]])
        z2=m.forward([[0.3]])
        q2=resultgenerator([[0.3]])
        z3=m.forward([[-0.3]])   
        q3=resultgenerator([[-0.3]])    
        print(q1[0][0]-z1[0][0], q2[0][0]-z2[0][0],q3[0][0]-z3[0][0],sum([ i[0]for i in loss])/batch)

    # for i in m.layer:
        # print(i)
 