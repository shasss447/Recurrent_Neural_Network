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

        for _,x in enumerate(inputs):
            h=np.tanh(self.Wxh@x+self.Whh@h+self.bxh)
        y=self.Wyh@h+self.byh
        
        return y,h