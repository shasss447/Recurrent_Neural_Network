import numpy as np
from numpy.random import randn

class RNN:

    def __init__(self, i_sz,o_sz,h_sz=64):
        self.Wxh=randn(h_sz,i_sz)/1000
        self.Whh=randn(h_sz,h_sz)/1000
        self.Wyh=randn(o_sz,h_sz)/1000

        self.bxh=np.zeros((h_sz,1))
        self.byh=np.zeros((o_sz,1))

    def forward(self,inputs):
        h=np.zeros((self.Whh.shape[0],1))

        self.ls_in=inputs
        self.ls_h={0:h}

        for i,x in enumerate(inputs):
            h=np.tanh(self.Wxh@x+self.Whh@h+self.bxh)
            self.ls_h[i+1]=h

        y=self.Wyh@h+self.byh

        return y,h
    
    def backprop(self,dy,lr=2e-2):
        n=len(self.ls_in)

        d_why=dy@self.ls_h[n].T
        d_byh=dy

        d_whh=np.zeros(self.Whh.shape)
        d_wxh=np.zeros(self.Wxh.shape)
        d_bxh=np.zeros(self.bxh.shape)

        dh=self.Wyh.T@dy
        
        for t in reversed(range(n)):
            tp=((1-self.ls_h[t+1]**2)*dh)
            d_bxh+=tp
            d_whh+=tp@self.ls_h[t].T
            d_wxh+=tp@self.ls_in[t].T
            dh=self.Whh.T@tp
        
        for d in [d_wxh,d_whh,d_why,d_bxh,d_byh]:
            np.clip(d,-1,1,out=d)

        self.Whh-=lr*d_whh
        self.Wxh-=lr*d_wxh
        self.Wyh-=lr*d_why
        self.bxh-=lr*d_bxh
        self.byh-=lr*d_byh